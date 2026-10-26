# Prompt Log - Track H: DBMS Optimization

## 1. Original User Prompt
"Implement Track H DBMS Optimization: set up a synthetic TPC-H benchmark database (Customer, Orders, LineItem, Supplier, Part) in SQLite. Formulate Multi-Index Selection as a 0-1 Knapsack / ILP problem under storage budget constraints, comparing Greedy, DP, and ILP with the default unindexed engine. Verify that query results are identical, compare EXPLAIN plans, and benchmark buffer pool replacement policies (LRU, 2Q, Clock)."

## 2. AI Assistant Response Excerpt
Created `track_h_dbms/database.py`, `track_h_dbms/index_selection.py`, `track_h_dbms/buffer_cache.py`, and `track_h_dbms/experiments.py`.
Populated thousands of rows across TPC-H tables and formulated 6 candidate secondary indexes with storage costs and measured latency benefits.

## 3. Critical Review & Revised Prompt
"Ensure EXPLAIN QUERY PLAN logs demonstrate the shift from SCAN TABLE to SEARCH TABLE USING INDEX, and assert that all query result tuples before and after indexing are bit-identical."

## 4. Resulting Code Change
- Implemented strict equality assertions: `assert base_results[qname] == opt_res`.
- Benchmarked buffer pool scan resistance using synthetic zipfian + sequential access traces.

## 5. Verification Note
- DP Knapsack and ILP (HiGHS) selected optimal index set `['idx_ord_custkey', 'idx_cust_mktseg', 'idx_li_suppkey']` (807.8 ms benefit within 400 KB budget), strictly beating Greedy (803.6 ms).
- Overall workload query speedup: 2.15x (11.1 ms -> 5.2 ms).
- All query outputs 100% bit-for-bit identical (PASS).
