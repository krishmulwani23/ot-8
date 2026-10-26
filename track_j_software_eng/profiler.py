"""
Codebase Profiler using cProfile (Track J).
Profiles the test suite execution pipeline on the workflow engine codebase.
"""

import cProfile
import pstats
import io
from typing import List, Dict, Any
from track_j_software_eng.workflow_engine.test_suite import get_all_test_cases

class CodebaseProfiler:
    @staticmethod
    def profile_full_suite() -> Dict[str, Any]:
        tests = get_all_test_cases()
        pr = cProfile.Profile()
        pr.enable()

        # Run all test functions
        for t in tests:
            t.fn()

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(15)

        profile_text = s.getvalue()
        # Parse top bottleneck function entries
        lines = profile_text.splitlines()
        top_functions = []
        for line in lines:
            if "workflow_engine" in line or "threading" in line or "time.sleep" in line:
                top_functions.append(line.strip())

        return {
            "total_tests_run": len(tests),
            "profile_summary": profile_text[:800],
            "top_bottlenecks": top_functions[:6]
        }
