#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_reports_to_json_gemini.py
- Đọc tất cả report trong ./reports/**/(report.txt | partial_report.txt)
- Gọi Gemini 2.5 Pro với system prompt ép JSON có bullets + citations [n]
- Chuẩn hoá, validate:
    * Bullet phải kết thúc bằng " [n]" hoặc " [n, m]"
    * Không cho [n/a], id phải là số dương
    * Cắt bỏ citations không dùng (nếu PRUNE_UNUSED_CITATIONS=True)
- Ghi answers.json. Có thống kê pass/fail + cảnh báo nguồn yếu.

YÊU CẦU:
pip install google-generativeai
"""

import os
import sys
import json
import glob
import asyncio
import random
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai

# ========= CẤU HÌNH =========
REPORTS_ROOT = Path("reports")
OUTPUT_JSON  = Path("answers.json")
MODEL        = "gemini-2.5-pro"

# Song song / retry
MAX_CONCURRENCY = 4
TIMEOUT_SEC  = 150
RETRIES      = 2
BACKOFF_BASE = 1.6

# Input quá dài thì cắt (0 = tắt)
MAX_REPORT_CHARS = 0  # ví dụ: 120_000

# Nhúng API key cho dễ test (ưu tiên ENV trước, sau đó dùng default)
GOOGLE_API_KEY_DEFAULT = "api"

# Kiểm soát hành vi validator
STRICT_END_INDEX = True               # yêu cầu bullet phải KẾT THÚC bằng [...]. Nếu False, chỉ cần có [n] ở đâu đó.
PRUNE_UNUSED_CITATIONS = True         # tự cắt citations không được tham chiếu
FAIL_ON_INVALID_IDS = False           # nếu True -> coi invalid IDs là lỗi cứng, không chỉ cảnh báo
WARN_WEAK_SOURCES = True              # cảnh báo nguồn "yếu"

SYSTEM_PROMPT = """Bạn là một chuyên gia phân tích thông tin. Nhiệm vụ: đọc một [Báo cáo] và trả lời chính xác, cô đọng cho [Câu hỏi] duy nhất dựa DUY NHẤT vào nội dung [Báo cáo].

ĐẦU VÀO:
- [Câu hỏi]: chuỗi văn bản
- [Báo cáo]: toàn bộ text của báo cáo, có thể kèm phần References với các chỉ số [n].

ĐẦU RA (BẮT BUỘC JSON, KHÔNG THÊM VĂN BẢN NGOÀI JSON):
{
  "question": "string",
  "verdict": "Có | Không | Không có đủ thông tin",
  "detail_bullets": [
    "mỗi bullet là một mệnh đề sự thật trực tiếp, KHÔNG dùng các cụm như 'Báo cáo cho biết…'; PHẢI kết thúc bằng khoảng trắng rồi tới chỉ mục dạng [n] hoặc [n, m, ...], ví dụ: 'Phí trung bình dưới 0,04 USD. [2, 5]'"
  ],
  "citations": [
    {"id": 1, "url": "https://..."},
    {"id": 2, "url": "https://..."}
  ]
}

QUY TẮC TRÍCH DẪN:
1) KHÔNG bịa nguồn. Chỉ sử dụng URL/nguồn thực sự xuất hiện trong [Báo cáo] (kể cả phần 'References' nếu có).
2) Nếu [Báo cáo] có sẵn chỉ mục [n] → URL trong phần References, HÃY TÁI SỬ DỤNG đúng số đó trong 'citations' và trong 'detail_bullets'.
3) Nếu [Báo cáo] KHÔNG có chỉ mục sẵn, hãy tự ĐÁNH SỐ MỚI bắt đầu từ 1 theo thứ tự xuất hiện, và dùng NHẤT QUÁN giữa 'detail_bullets' và 'citations'.
4) Mỗi bullet trong 'detail_bullets' PHẢI kết thúc bằng ' [..]' chứa ít nhất 1 id hợp lệ; ví dụ: '... [3]' hoặc '... [1, 4]'.
5) 'citations' là danh sách {id,url}; id là số nguyên dương. Chỉ URL trực tiếp, không mô tả thêm.

LƯU Ý:
- Nếu không đủ dữ kiện trong [Báo cáo], đặt 'verdict' = 'Không có đủ thông tin' và chỉ tạo các bullet nêu rõ thiếu dữ kiện (cũng có [n] nếu có nguồn nói thiếu).
- Xuất ra JSON-OBJECT hợp lệ duy nhất, không có bất kỳ bình luận hay text ngoài JSON.
"""

# ========= HỖ TRỢ I/O =========

@dataclass
class UnitResult:
    report_dir: str
    payload: Optional[Dict[str, Any]]
    error: Optional[str] = None
    warnings: List[str] = None

def _read_txt(p: Path) -> Optional[str]:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return None

def _truncate_report_if_needed(txt: str) -> str:
    if MAX_REPORT_CHARS and len(txt) > MAX_REPORT_CHARS:
        return txt[:MAX_REPORT_CHARS] + "\n\n[TRUNCATED]\n"
    return txt

def _detect_question(report_dir: Path, fallback: str = "Câu hỏi mặc định") -> str:
    meta = _read_txt(report_dir / "meta.txt")
    if meta:
        for line in meta.splitlines():
            if line.lower().startswith("query:"):
                return line.split(":", 1)[1].strip()
    qtxt = _read_txt(report_dir / "question.txt")
    if qtxt and qtxt.strip():
        return qtxt.strip()
    return fallback

# ========= GEMINI =========

def _configure_gemini():
    key = ""
    if not key:
        key = GOOGLE_API_KEY_DEFAULT.strip()
    if not key:
        print("Thiếu GOOGLE_API_KEY (env var) và GOOGLE_API_KEY_DEFAULT rỗng.")
        sys.exit(1)
    genai.configure(api_key=key)

def _build_model():
    return genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

def _extract_json_from_gemini_response(resp) -> str:
    text = getattr(resp, "text", None)
    if text and text.strip():
        return text.strip()
    try:
        d = resp.to_dict()
        cands = d.get("candidates", [])
        chunks = []
        for c in cands:
            content = c.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                t = part.get("text")
                if t:
                    chunks.append(t)
        text2 = "".join(chunks).strip()
        if text2:
            return text2
    except Exception:
        pass
    return ""

async def _call_gemini_json(model, question: str, report_text: str, timeout: int = TIMEOUT_SEC) -> Dict[str, Any]:
    payload = {"Câu hỏi": question, "Báo cáo": report_text}
    prompt = json.dumps(payload, ensure_ascii=False)

    def _sync_call():
        return model.generate_content(contents=[{"role": "user", "parts": [prompt]}])

    resp = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=timeout)
    text = _extract_json_from_gemini_response(resp)
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))

# ========= CHUẨN HOÁ + VALIDATE =========

ID_LIST_RE = re.compile(r"\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]")

WEAK_SOURCE_PATTERNS = [
    r"scribd\.com", r"coursehero\.com", r"medium\.com", r"blogspot\.",
    r"wordpress\.", r"b2binpay\.com", r"coincodex\.com/blog", r"reddit\.com",
]

def _is_weak_source(url: str) -> bool:
    u = url.lower()
    return any(re.search(p, u) for p in WEAK_SOURCE_PATTERNS)

def _normalize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    q = str(data.get("question", "")).strip()
    v = str(data.get("verdict", "")).strip()

    bullets = data.get("detail_bullets") or []
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    bullets = [str(x).strip() for x in bullets if str(x).strip()]

    cites = data.get("citations") or []
    norm_cites: List[Dict[str, Any]] = []
    if isinstance(cites, list):
        for it in cites:
            if isinstance(it, dict) and "id" in it and "url" in it:
                try:
                    cid = int(it["id"])
                    curl = str(it["url"]).strip()
                    if cid > 0 and curl:
                        norm_cites.append({"id": cid, "url": curl})
                except Exception:
                    continue

    # Dedup by id (keep last url)
    id_to_url = {}
    for c in norm_cites:
        id_to_url[c["id"]] = c["url"]
    norm_cites = [{"id": k, "url": id_to_url[k]} for k in sorted(id_to_url.keys())]

    return {
        "question": q,
        "verdict": v,
        "detail_bullets": bullets,
        "citations": norm_cites,
    }

def _collect_used_ids_from_bullets(bullets: List[str]) -> Tuple[set, List[str]]:
    used = set()
    errs = []
    for i, b in enumerate(bullets, 1):
        if STRICT_END_INDEX:
            if not b.endswith("]"):
                errs.append(f"Bullet #{i} không kết thúc bằng chỉ mục [..].")
        # tìm mọi [..] trong bullet
        matches = list(ID_LIST_RE.finditer(b))
        if not matches:
            errs.append(f"Bullet #{i} không có chỉ mục [n].")
            continue
        # lấy block cuối cùng (thường là ở cuối câu)
        ids_str = matches[-1].group(1)
        for num in re.split(r"\s*,\s*", ids_str.strip()):
            if not re.fullmatch(r"\d+", num):
                errs.append(f"Bullet #{i} chứa chỉ mục không hợp lệ: '{num}'.")
                continue
            used.add(int(num))
    return used, errs

def _validate_and_prune(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Trả về (payload_sau_khi_prune, errors, warnings)
    - errors: lỗi format (thiếu chỉ mục, id không tồn tại…)
    - warnings: cảnh báo nguồn yếu, citations dư thừa…
    """
    bullets = payload.get("detail_bullets") or []
    cites = payload.get("citations") or []
    errors: List[str] = []
    warnings: List[str] = []

    # Check chữ "n/a" trong bullet index → lỗi
    for i, b in enumerate(bullets, 1):
        if re.search(r"\[(?:\s*n/?a\s*)\]", b, flags=re.IGNORECASE):
            errors.append(f"Bullet #{i} dùng [n/a] (không hợp lệ).")

    used_ids, errs_bullets = _collect_used_ids_from_bullets(bullets)
    errors.extend(errs_bullets)

    id_set = {c["id"] for c in cites if isinstance(c, dict) and "id" in c}
    # id trong bullets nhưng không có trong citations
    for nid in sorted(used_ids):
        if nid not in id_set:
            errors.append(f"Tham chiếu id [{nid}] không tồn tại trong citations.")

    # citations không dùng
    unused = sorted(list(id_set - used_ids))
    if unused:
        msg = f"Citations không được tham chiếu: {unused}"
        if PRUNE_UNUSED_CITATIONS:
            # prune
            cites = [c for c in cites if c["id"] in used_ids]
            payload["citations"] = cites
            msg += " → ĐÃ CẮT BỎ."
        warnings.append(msg)

    # cảnh báo nguồn yếu
    if WARN_WEAK_SOURCES:
        weak = [c for c in cites if _is_weak_source(c["url"])]
        if weak:
            warnings.append("Nguồn có độ tin cậy thấp: " + ", ".join(f'[{c["id"]}] {c["url"]}' for c in weak))

    return payload, errors, warnings

# ========= PIPELINE =========

async def _process_one(model, report_txt: Path) -> UnitResult:
    report_dir = report_txt.parent
    text = _read_txt(report_txt) or _read_txt(report_dir / "partial_report.txt")
    if not text or not text.strip():
        return UnitResult(str(report_dir), None, "Report rỗng hoặc không tồn tại", [])

    question = _detect_question(report_dir)
    text = _truncate_report_if_needed(text)

    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            data = await _call_gemini_json(model, question, text, timeout=TIMEOUT_SEC)
            payload = _normalize_payload(data)
            payload, errs, warns = _validate_and_prune(payload)

            # nếu bật FAIL_ON_INVALID_IDS và có lỗi → trả lỗi
            if errs and FAIL_ON_INVALID_IDS:
                return UnitResult(str(report_dir), None, "; ".join(errs), warns)

            # gắn lỗi mềm trong field error để bạn xem lại nhưng vẫn xuất payload
            soft_error = "; ".join(errs) if errs else None
            return UnitResult(str(report_dir), payload, soft_error, warns)
        except asyncio.TimeoutError:
            last_err = f"Timeout sau {TIMEOUT_SEC}s (attempt {attempt+1}/{RETRIES+1})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        if attempt < RETRIES:
            sleep_s = (BACKOFF_BASE ** attempt) + random.uniform(0, 0.6)
            print(f"[RETRY] {report_dir.name}: {last_err} -> thử lại sau {sleep_s:.2f}s")
            await asyncio.sleep(sleep_s)
        else:
            return UnitResult(str(report_dir), None, last_err, [])

async def main():
    _configure_gemini()
    model = _build_model()

    # tìm report.txt
    report_files = [Path(p) for p in glob.glob(str(REPORTS_ROOT / "**" / "report.txt"), recursive=True)]
    # thêm partial-only
    partial_only = [
        Path(p) for p in glob.glob(str(REPORTS_ROOT / "**" / "partial_report.txt"), recursive=True)
        if not (Path(p).parent / "report.txt").exists()
    ]
    report_files.extend([p.parent / "report.txt" for p in partial_only])

    if not report_files:
        print("Không tìm thấy báo cáo nào trong 'reports/'.")
        sys.exit(1)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(path: Path):
        async with sem:
            print(f"[AGENT] {path.parent}")
            return await _process_one(model, path)

    results: List[UnitResult] = await asyncio.gather(*[asyncio.create_task(worker(p)) for p in report_files])

    out: List[Dict[str, Any]] = []
    ok_count = 0
    warn_count = 0
    err_count = 0

    for r in results:
        if r.payload:
            item = {
                "report_dir": r.report_dir,
                **r.payload,     # question, verdict, detail_bullets, citations
                "error": r.error,
            }
            # nối warnings (nếu có)
            if r.warnings:
                item["warnings"] = r.warnings
                warn_count += len(r.warnings)
            out.append(item)
            if not r.error:
                ok_count += 1
            else:
                err_count += 1
        else:
            out.append({
                "report_dir": r.report_dir,
                "question": _detect_question(Path(r.report_dir)),
                "verdict": "Không có đủ thông tin",
                "detail_bullets": [],
                "citations": [],
                "error": r.error or "Unknown error"
            })
            err_count += 1

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"Total items     : {len(out)}")
    print(f"Valid (no error): {ok_count}")
    print(f"Soft errors     : {err_count}")
    print(f"Warnings        : {warn_count}")
    print(f"JSON output     : {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBị hủy bởi người dùng.")
        sys.exit(130)