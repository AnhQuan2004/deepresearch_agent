"""
Example usage of the DeepResearcher to produce a report.

See deep_output.txt for the console output from running this script, and deep_output.pdf for the final report
"""

import asyncio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deep_researcher import DeepResearcher

manager = DeepResearcher(
    max_iterations=3,
    max_time_minutes=5,
    verbose=True,
    tracing=True
)

# query = "Write a report on Plato - who was he, what were his main works " \
#         "and what are the main philosophical ideas he's known for"

query = "Tốc độ tăng trưởng kép hàng năm (CAGR) của lĩnh vực dự án BNB hoạt động có được dự báo > 20% không?CAGR chung của blockchain, dApp, Smart Contract (Grandview),..."

report = asyncio.run(
    manager.run(
        query
    )
)

print("\n=== Final Report ===")
print(report)

save_path = "deep_output.txt"
with open(save_path, "w") as f:
    f.write(report)
print(f"Report saved to {save_path}")