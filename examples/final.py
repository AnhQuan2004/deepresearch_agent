import os
import sys
import re
import json
import glob
import asyncio
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# === đảm bảo import được deep_researcher ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deep_researcher import DeepResearcher

# ============ CẤU HÌNH CHUNG ============
# Multi-query
MAX_CONCURRENCY = 4          # số query chạy song song tối đa
PER_QUERY_TIMEOUT_MIN = 100  # timeout mỗi query (phút)
OUTPUT_ROOT = Path("reports")

# Aggregation -> JSON
OUTPUT_JSON = Path("answers.json")
GEMINI_MODEL = "gemini-2.5-pro"

# Google Generative AI API Key (ưu tiên ENV rồi tới DEFAULT)
GOOGLE_API_KEY_DEFAULT = ""  # Điền key của bạn ở đây nếu không dùng env

# File chứa queries (mỗi dòng là 1 query)
QUERIES_FILE = Path(__file__).parent / "queries.txt"

# Validator & xử lý
STRICT_END_INDEX = True        # bullet PHẢI kết thúc bằng " [n]" hoặc " [n, m]"
PRUNE_UNUSED_CITATIONS = True  # tự cắt citations không dùng
FAIL_ON_INVALID_IDS = False    # True -> fail cứng nếu lỗi format
WARN_WEAK_SOURCES = True       # cảnh báo nguồn yếu
MAX_REPORT_CHARS = 0           # 0 = không cắt; ví dụ 120_000 nếu report cực dài

# ============ System prompt cho agent ============
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

# ============ Utility Functions ============

def slugify(text: str, max_len: int = 60) -> str:
    """Chuyển text thành slug an toàn cho tên thư mục"""
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = text.replace(" ", "-")
    return text[:max_len] if len(text) > max_len else text

def load_queries_from_file(file_path: Path) -> List[str]:
    """Đọc queries từ file, mỗi dòng là 1 query"""
    queries = []
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Bỏ qua dòng trống và comment (bắt đầu bằng #)
                    if line and not line.startswith("#"):
                        queries.append(line)
            print(f"Đã load {len(queries)} queries từ {file_path}")
        except Exception as e:
            print(f"Lỗi khi đọc file queries: {e}")
    else:
        print(f"File {file_path} không tồn tại")
    return queries

# ============ Multi Query (DeepResearcher) ============

def make_manager():
    """Tạo instance mới của DeepResearcher cho mỗi query"""
    return DeepResearcher(
        max_iterations=3,
        max_time_minutes=PER_QUERY_TIMEOUT_MIN - 1,  # để nội bộ dừng trước timeout cứng
        verbose=True,
        tracing=True
    )

async def run_one_query(query: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    """Chạy một query với DeepResearcher"""
    slug = slugify(query)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_ROOT / f"{ts}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.txt"
    meta_path = out_dir / "meta.txt"

    # Ghi meta ban đầu
    meta = [
        f"query: {query}",
        f"slug: {slug}",
        f"time: {ts}",
        f"started: {datetime.now().isoformat()}"
    ]
    meta_path.write_text("\n".join(meta), encoding="utf-8")

    manager = make_manager()
    print(f"[START] {slug[:40]}...")
    
    async with sem:
        try:
            timeout = PER_QUERY_TIMEOUT_MIN * 60
            report = await asyncio.wait_for(manager.run(query), timeout=timeout)

            # Lưu report
            report_path.write_text(report, encoding="utf-8")
            
            # Update meta
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"\nlen_chars: {len(report)}")
                f.write(f"\nstatus: success")
                f.write(f"\ncompleted: {datetime.now().isoformat()}\n")

            print(f"[DONE ] {slug[:40]}... -> {report_path}")
            return {
                "query": query,
                "slug": slug,
                "dir": str(out_dir),
                "ok": True,
                "error": None
            }
            
        except asyncio.TimeoutError:
            err = f"Timeout sau {PER_QUERY_TIMEOUT_MIN} phút"
            print(f"[FAIL ] {slug[:40]}... -> {err}")
            
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"\nerror: {err}")
                f.write(f"\nstatus: timeout")
                f.write(f"\nfailed: {datetime.now().isoformat()}\n")
            
            # Lưu kết quả partial nếu có
            try:
                if hasattr(manager, "_last_partial_result") and manager._last_partial_result:
                    partial_path = out_dir / "partial_report.txt"
                    partial_path.write_text(manager._last_partial_result, encoding="utf-8")
                    print(f"[INFO ] Lưu kết quả một phần vào {partial_path}")
            except Exception as e2:
                print(f"[WARN ] Không thể lưu partial: {e2}")
                
            return {
                "query": query,
                "slug": slug,
                "dir": str(out_dir),
                "ok": False,
                "error": err
            }
            
        except Exception as e:
            err = str(e)
            print(f"[FAIL ] {slug[:40]}... -> {err}")
            
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"\nerror: {err}")
                f.write(f"\nstatus: error")
                f.write(f"\nfailed: {datetime.now().isoformat()}\n")
            
            # Lưu kết quả partial nếu có
            try:
                if hasattr(manager, "_last_partial_result") and manager._last_partial_result:
                    partial_path = out_dir / "partial_report.txt"
                    partial_path.write_text(manager._last_partial_result, encoding="utf-8")
                    print(f"[INFO ] Lưu kết quả một phần vào {partial_path}")
            except Exception as e2:
                print(f"[WARN ] Không thể lưu partial: {e2}")
                
            return {
                "query": query,
                "slug": slug,
                "dir": str(out_dir),
                "ok": False,
                "error": err
            }

async def run_many_queries(queries: List[str]) -> List[Dict[str, Any]]:
    """Chạy nhiều queries song song"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # Chạy các tasks
    tasks = [run_one_query(q, sem) for q in queries]
    results = await asyncio.gather(*tasks)

    # Tạo index cho batch này
    index_path = OUTPUT_ROOT / "index.md"
    lines = [
        "# Reports Index",
        f"\nGenerated: {datetime.now().isoformat()}\n"
    ]
    
    for r in results:
        status = "✅" if r["ok"] else "❌"
        rel = os.path.relpath(r["dir"], OUTPUT_ROOT)
        lines.append(f"- {status} `{r['slug']}` — [{rel}](./{rel})")
        lines.append(f"  - Query: {r['query'][:100]}...")
        if r["error"]:
            lines.append(f"  - Error: {r['error']}")
    
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok_count = sum(1 for r in results if r["ok"])
    print(f"\nTổng kết DeepResearcher: {ok_count}/{len(results)} thành công.")
    print(f"Mục lục: {index_path}\n")
    
    return results

# ============ Agent (Gemini 2.5 Pro) ============

import google.generativeai as genai

def configure_gemini():
    """Cấu hình Gemini API"""
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        key = GOOGLE_API_KEY_DEFAULT.strip()
    if not key:
        print("❌ Lỗi: Thiếu GOOGLE_API_KEY. Vui lòng:")
        print("   - Set biến môi trường GOOGLE_API_KEY, hoặc")
        print("   - Điền vào GOOGLE_API_KEY_DEFAULT trong file")
        sys.exit(1)
    genai.configure(api_key=key)
    print("✅ Đã cấu hình Gemini API")

def build_model():
    """Tạo Gemini model với cấu hình JSON response"""
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

def _read_txt(p: Path) -> Optional[str]:
    """Đọc file text an toàn"""
    try:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Không thể đọc {p}: {e}")
    return None

def _truncate_report_if_needed(txt: str) -> str:
    """Cắt report nếu quá dài"""
    if MAX_REPORT_CHARS > 0 and len(txt) > MAX_REPORT_CHARS:
        return txt[:MAX_REPORT_CHARS] + "\n\n[TRUNCATED]\n"
    return txt

def detect_question(report_dir: Path, fallback: str = "Câu hỏi mặc định") -> str:
    """Phát hiện câu hỏi từ meta hoặc question.txt"""
    # Ưu tiên lấy từ meta.txt
    meta = _read_txt(report_dir / "meta.txt")
    if meta:
        for line in meta.splitlines():
            if line.lower().startswith("query:"):
                query = line.split(":", 1)[1].strip()
                if query:
                    return query
    
    # Fallback sang question.txt
    qtxt = _read_txt(report_dir / "question.txt")
    if qtxt and qtxt.strip():
        return qtxt.strip()
    
    return fallback

def _extract_json_from_response(resp) -> str:
    """Extract JSON text từ Gemini response"""
    # Cách 1: Lấy từ text attribute
    text = getattr(resp, "text", None)
    if text and text.strip():
        return text.strip()
    
    # Cách 2: Parse từ response dict
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

async def call_gemini_json(model, question: str, report_text: str, timeout_sec: int = 150) -> Dict[str, Any]:
    """Gọi Gemini và parse JSON response"""
    payload = {
        "Câu hỏi": question,
        "Báo cáo": report_text
    }
    prompt = json.dumps(payload, ensure_ascii=False)

    def _sync_call():
        return model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}]
        )

    resp = await asyncio.wait_for(
        asyncio.to_thread(_sync_call),
        timeout=timeout_sec
    )
    
    text = _extract_json_from_response(resp)
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    
    # Parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Thử extract JSON object từ text
        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))

# ============ Validate & Normalize ============

ID_LIST_RE = re.compile(r"\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]")

WEAK_SOURCE_PATTERNS = [
    r"scribd\.com",
    r"coursehero\.com",
    r"medium\.com",
    r"blogspot\.",
    r"wordpress\.",
    r"b2binpay\.com",
    r"coincodex\.com/blog",
    r"reddit\.com",
]

def is_weak_source(url: str) -> bool:
    """Kiểm tra URL có phải nguồn yếu không"""
    u = url.lower()
    return any(re.search(p, u) for p in WEAK_SOURCE_PATTERNS)

def normalize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa payload từ Gemini"""
    q = str(data.get("question", "")).strip()
    v = str(data.get("verdict", "")).strip()
    
    # Normalize bullets
    bullets = data.get("detail_bullets") or []
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    bullets = [str(x).strip() for x in bullets if str(x).strip()]

    # Normalize citations
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

    # Dedup by id
    id_to_url = {}
    for c in norm_cites:
        id_to_url[c["id"]] = c["url"]
    norm_cites = [{"id": k, "url": id_to_url[k]} for k in sorted(id_to_url.keys())]

    return {
        "question": q,
        "verdict": v,
        "detail_bullets": bullets,
        "citations": norm_cites
    }

def collect_used_ids(bullets: List[str]) -> Tuple[set, List[str]]:
    """Thu thập các ID được sử dụng trong bullets"""
    used = set()
    errs = []
    
    for i, b in enumerate(bullets, 1):
        if STRICT_END_INDEX and not b.endswith("]"):
            errs.append(f"Bullet #{i} không kết thúc bằng chỉ mục [..]")
        
        matches = list(ID_LIST_RE.finditer(b))
        if not matches:
            errs.append(f"Bullet #{i} không có chỉ mục [n]")
            continue
        
        # Lấy match cuối cùng (thường là citation)
        ids_str = matches[-1].group(1)
        for num in re.split(r"\s*,\s*", ids_str.strip()):
            if not re.fullmatch(r"\d+", num):
                errs.append(f"Bullet #{i} chứa chỉ mục không hợp lệ: '{num}'")
                continue
            used.add(int(num))
    
    return used, errs

def validate_and_prune(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Validate và prune citations không dùng"""
    bullets = payload.get("detail_bullets") or []
    cites = payload.get("citations") or []
    errors: List[str] = []
    warnings: List[str] = []

    # Kiểm tra [n/a]
    for i, b in enumerate(bullets, 1):
        if re.search(r"\[(?:\s*n/?a\s*)\]", b, flags=re.IGNORECASE):
            errors.append(f"Bullet #{i} dùng [n/a] (không hợp lệ)")

    # Thu thập used IDs
    used_ids, errs_b = collect_used_ids(bullets)
    errors.extend(errs_b)

    # Kiểm tra tham chiếu
    id_set = {c["id"] for c in cites if isinstance(c, dict) and "id" in c}
    for nid in sorted(used_ids):
        if nid not in id_set:
            errors.append(f"Tham chiếu id [{nid}] không tồn tại trong citations")

    # Prune citations không dùng
    unused = sorted(list(id_set - used_ids))
    if unused:
        msg = f"Citations không được tham chiếu: {unused}"
        if PRUNE_UNUSED_CITATIONS:
            cites = [c for c in cites if c["id"] in used_ids]
            payload["citations"] = cites
            msg += " → ĐÃ CẮT BỎ"
        warnings.append(msg)

    # Cảnh báo nguồn yếu
    if WARN_WEAK_SOURCES:
        weak = [c for c in cites if is_weak_source(c["url"])]
        if weak:
            weak_msg = "Nguồn có độ tin cậy thấp: "
            weak_details = [f'[{c["id"]}] {c["url"][:50]}...' for c in weak]
            warnings.append(weak_msg + ", ".join(weak_details))

    return payload, errors, warnings

# ============ Orchestrator ============

@dataclass
class AgentUnit:
    report_dir: str
    payload: Optional[Dict[str, Any]]
    error: Optional[str]
    warnings: List[str]

async def aggregate_dirs_to_json(dirs: List[str]) -> List[AgentUnit]:
    """Aggregate các report thành JSON với Gemini"""
    configure_gemini()
    model = build_model()

    async def process_dir(d: str) -> AgentUnit:
        report_dir = Path(d)
        
        # Đọc report (ưu tiên report.txt, fallback sang partial_report.txt)
        report_text = _read_txt(report_dir / "report.txt")
        if not report_text or not report_text.strip():
            report_text = _read_txt(report_dir / "partial_report.txt")
        
        if not report_text or not report_text.strip():
            return AgentUnit(d, None, "Report rỗng hoặc không tồn tại", [])

        question = detect_question(report_dir)
        report_text = _truncate_report_if_needed(report_text)

        last_err = None
        for attempt in range(3):
            try:
                data = await call_gemini_json(model, question, report_text, timeout_sec=150)
                payload = normalize_payload(data)
                payload, errs, warns = validate_and_prune(payload)

                if errs and FAIL_ON_INVALID_IDS:
                    return AgentUnit(d, None, "; ".join(errs), warns)

                soft_error = "; ".join(errs) if errs else None
                return AgentUnit(d, payload, soft_error, warns)
                
            except asyncio.TimeoutError:
                last_err = f"Timeout khi gọi Gemini (attempt {attempt+1}/3)"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

            # Backoff ngắn
            await asyncio.sleep((1.6 ** attempt) + random.uniform(0, 0.5))

        return AgentUnit(d, None, last_err or "Gemini error", [])

    # Process song song với semaphore
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async def wrapped(d):
        async with sem:
            print(f"[AGENT] Processing {Path(d).name}")
            return await process_dir(d)

    tasks = [asyncio.create_task(wrapped(d)) for d in dirs]
    results = await asyncio.gather(*tasks)
    return results

# ============ Main Function ============

async def main():
    """Main orchestrator function"""
    
    # 1) Load queries
    queries = load_queries_from_file(QUERIES_FILE)
    
    # Nếu không có queries từ file, dùng default
    if not queries:
        print("⚠️ Không có queries trong file, sử dụng queries mặc định")
        queries = [
            "Tốc độ tăng trưởng kép hàng năm (CAGR) của lĩnh vực dự án bnb hoạt động có được dự báo > 20% không?"
        ]
    
    print(f"\n📋 Số lượng queries: {len(queries)}")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q[:80]}...")

    # 2) Chạy DeepResearcher song song
    print("\n" + "="*60)
    print("🚀 PHASE 1: RUN MULTI-QUERY (DeepResearcher)")
    print("="*60)
    
    results = await run_many_queries(queries)

    # Lấy danh sách thư mục có report
    dirs_for_agent: List[str] = []
    for r in results:
        d = Path(r["dir"])
        if (d / "report.txt").exists() or (d / "partial_report.txt").exists():
            dirs_for_agent.append(r["dir"])

    if not dirs_for_agent:
        print("❌ Không có report nào để tổng hợp.")
        return

    # 3) Gọi agent (Gemini) để tổng hợp
    print("\n" + "="*60)
    print("🤖 PHASE 2: RUN AGENT (Gemini)")
    print("="*60)
    
    agent_units = await aggregate_dirs_to_json(dirs_for_agent)

    # 4) Tổng hợp kết quả
    out: List[Dict[str, Any]] = []
    ok_count = 0
    warn_count = 0
    err_count = 0

    for u in agent_units:
        if u.payload:
            item = {
                "report_dir": u.report_dir,
                **u.payload,
                "error": u.error
            }
            if u.warnings:
                item["warnings"] = u.warnings
                warn_count += len(u.warnings)
            out.append(item)
            
            if not u.error:
                ok_count += 1
            else:
                err_count += 1
        else:
            # Fallback cho case lỗi
            item = {
                "report_dir": u.report_dir,
                "question": detect_question(Path(u.report_dir)),
                "verdict": "Không có đủ thông tin",
                "detail_bullets": [],
                "citations": [],
                "error": u.error or "Unknown error"
            }
            out.append(item)
            err_count += 1

    # 5) Ghi JSON output
    OUTPUT_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 6) In summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"📁 Batch reports   : {len(dirs_for_agent)}")
    print(f"✅ Valid (no error): {ok_count}")
    print(f"⚠️  Soft errors     : {err_count}")
    print(f"⚡ Warnings        : {warn_count}")
    print(f"💾 JSON output     : {OUTPUT_JSON}")
    print("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Bị hủy bởi người dùng.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
