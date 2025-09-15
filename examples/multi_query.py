import asyncio
import os
import sys
import re
from datetime import datetime
from pathlib import Path

# đảm bảo import được deep_researcher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deep_researcher import DeepResearcher  # noqa

# ==== cấu hình ====
MAX_CONCURRENCY = 4         # số query chạy song song tối đa
PER_QUERY_TIMEOUT_MIN = 100   # timeout mỗi query (phút)
OUTPUT_ROOT = Path("reports")

def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = text.replace(" ", "-")
    return text[:max_len] if len(text) > max_len else text

def make_manager():
    # tạo mới cho mỗi query để tránh share state
    return DeepResearcher(
        max_iterations=3,
        max_time_minutes=PER_QUERY_TIMEOUT_MIN - 1,  # Giảm 1 phút để đảm bảo timeout nội bộ xảy ra trước timeout bên ngoài
        verbose=True,
        tracing=True
    )

async def run_one(query: str, sem: asyncio.Semaphore):
    slug = slugify(query)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_ROOT / f"{ts}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.txt"
    meta_path = out_dir / "meta.txt"
    
    # Ghi meta ngay từ đầu để đảm bảo có thông tin ngay cả khi timeout
    meta = [
        f"query: {query}",
        f"slug: {slug}",
        f"time: {ts}",
    ]
    meta_path.write_text("\n".join(meta), encoding="utf-8")

    manager = make_manager()

    print(f"[START] {slug}")
    async with sem:
        try:
            # timeout cứng cho mỗi query
            timeout = PER_QUERY_TIMEOUT_MIN * 60
            report = await asyncio.wait_for(manager.run(query), timeout=timeout)

            # lưu report
            report_path.write_text(report, encoding="utf-8")

            # Cập nhật meta với thông tin thành công
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"\nlen_chars: {len(report)}\n")
                f.write(f"status: success\n")

            print(f"[DONE ] {slug} -> {report_path}")
            return {"query": query, "slug": slug, "dir": str(out_dir), "ok": True, "error": None}
        except asyncio.TimeoutError:
            err = f"Timeout sau {PER_QUERY_TIMEOUT_MIN} phút"
            print(f"[FAIL ] {slug} -> {err}")
            
            # Lưu thông tin timeout vào meta
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"\nerror: {err}\n")
                f.write(f"status: timeout\n")
            
            # Ghi partial report nếu có
            try:
                if hasattr(manager, "_last_partial_result") and manager._last_partial_result:
                    partial_path = out_dir / "partial_report.txt"
                    partial_path.write_text(manager._last_partial_result, encoding="utf-8")
                    print(f"[INFO ] {slug} -> Lưu kết quả một phần vào {partial_path}")
            except Exception as e:
                print(f"[WARN ] Không thể lưu kết quả một phần: {e}")
                
            return {"query": query, "slug": slug, "dir": str(out_dir), "ok": False, "error": err}
        except Exception as e:
            err = str(e)
            print(f"[FAIL ] {slug} -> {err}")
            
            # Lưu thông tin lỗi vào meta
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"\nerror: {err}\n")
                f.write(f"status: error\n")
                
            # Ghi partial report nếu có
            try:
                if hasattr(manager, "_last_partial_result") and manager._last_partial_result:
                    partial_path = out_dir / "partial_report.txt"
                    partial_path.write_text(manager._last_partial_result, encoding="utf-8")
                    print(f"[INFO ] {slug} -> Lưu kết quả một phần vào {partial_path}")
            except Exception as e2:
                print(f"[WARN ] Không thể lưu kết quả một phần: {e2}")
                
            return {"query": query, "slug": slug, "dir": str(out_dir), "ok": False, "error": err}

async def run_many(queries: list[str]):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    results = []
    # ưu tiên dùng TaskGroup (Py 3.11+); nếu bạn đang ở 3.10, thay bằng gather
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(q, sem)) for q in queries]
        # TaskGroup không trả list; gom từ tasks:
        # (khi thoát TaskGroup, tất cả task đã xong)
        for t in tasks:
            results.append(t.result())
    except Exception as e:
        # Chỉ xảy ra nếu TaskGroup raise; nhưng run_one đã nuốt lỗi rồi
        print("Unexpected TaskGroup error:", e)

    # tạo index.md
    index_path = OUTPUT_ROOT / "index.md"
    lines = ["# Reports Index\n"]
    for r in results:
        status = "✅" if r["ok"] else "❌"
        rel = os.path.relpath(r["dir"], OUTPUT_ROOT)
        lines.append(f"- {status} `{r['slug']}` — [{rel}](./{rel})")
        if r["error"]:
            lines.append(f"  - error: {r['error']}")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nTổng kết: {sum(1 for r in results if r['ok'])}/{len(results)} thành công.")
    print(f"Xem mục lục tại: {index_path}")

    return results

if __name__ == "__main__":
    # == ví dụ danh sách queries ==
    queries = [
        "Tốc độ tăng trưởng kép hàng năm (CAGR) của lĩnh vực dự án bnb hoạt động có được dự báo > 20% không?CAGR chung của blockchain, dApp, Smart Contract (Grandview)",
        "Lịch trình trả token (vesting) cho đội ngũ và nhà đầu tư của bnb có được thiết kế để trả dần theo từng đợt nhỏ, thay vì mở khóa một lượng lớn tại một thời điểm không? (Linear)",
        "Giải pháp của dự án bnb có mang lại một lợi ích vượt trội ít nhất 5-10 lần (về tốc độ, chi phí, hoặc sự tiện lợi) so với giải pháp hiện có không?",
        "Thời gian để 50% token bnb được phát hành ban đầu có lớn hơn 2 năm không?",
    ]

    asyncio.run(run_many(queries))
