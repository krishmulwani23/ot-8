"""
Synthetic TPC-H Benchmark Schema and Data Generator (Track H).
Creates SQLite database with Customer, Orders, LineItem, Supplier, and Part tables.
"""

import sqlite3
import numpy as np
import time
from typing import Dict, List, Tuple, Any

class TPCHDatabase:
    def __init__(self, db_path: str = ":memory:", seed: int = 42):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.rng = np.random.default_rng(seed)
        self.create_schema()
        self.populate_data()

    def create_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
        DROP TABLE IF EXISTS lineitem;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS customer;
        DROP TABLE IF EXISTS supplier;
        DROP TABLE IF EXISTS part;

        CREATE TABLE customer (
            c_custkey INTEGER PRIMARY KEY,
            c_name TEXT NOT NULL,
            c_address TEXT NOT NULL,
            c_nationkey INTEGER NOT NULL,
            c_phone TEXT NOT NULL,
            c_acctbal REAL NOT NULL,
            c_mktsegment TEXT NOT NULL
        );

        CREATE TABLE orders (
            o_orderkey INTEGER PRIMARY KEY,
            o_custkey INTEGER NOT NULL,
            o_orderstatus TEXT NOT NULL,
            o_totalprice REAL NOT NULL,
            o_orderdate TEXT NOT NULL,
            o_orderpriority TEXT NOT NULL,
            o_clerk TEXT NOT NULL,
            o_shippriority INTEGER NOT NULL
        );

        CREATE TABLE lineitem (
            l_orderkey INTEGER NOT NULL,
            l_partkey INTEGER NOT NULL,
            l_suppkey INTEGER NOT NULL,
            l_linenumber INTEGER NOT NULL,
            l_quantity REAL NOT NULL,
            l_extendedprice REAL NOT NULL,
            l_discount REAL NOT NULL,
            l_tax REAL NOT NULL,
            l_returnflag TEXT NOT NULL,
            l_linestatus TEXT NOT NULL,
            l_shipdate TEXT NOT NULL,
            l_commitdate TEXT NOT NULL,
            l_receiptdate TEXT NOT NULL
        );

        CREATE TABLE supplier (
            s_suppkey INTEGER PRIMARY KEY,
            s_name TEXT NOT NULL,
            s_address TEXT NOT NULL,
            s_nationkey INTEGER NOT NULL,
            s_phone TEXT NOT NULL,
            s_acctbal REAL NOT NULL
        );

        CREATE TABLE part (
            p_partkey INTEGER PRIMARY KEY,
            p_name TEXT NOT NULL,
            p_mfgr TEXT NOT NULL,
            p_brand TEXT NOT NULL,
            p_type TEXT NOT NULL,
            p_size INTEGER NOT NULL,
            p_retailprice REAL NOT NULL
        );
        """)
        self.conn.commit()

    def populate_data(self, n_cust: int = 800, n_orders: int = 3000,
                      n_lines: int = 10000, n_supp: int = 150, n_part: int = 500):
        cur = self.conn.cursor()
        mktsegments = ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD']
        statuses = ['O', 'F', 'P']
        priorities = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']

        # 1. Customers
        cust_rows = []
        for i in range(1, n_cust + 1):
            cust_rows.append((
                i, f"Customer#{i:08d}", f"Address_{i}", int(self.rng.integers(0, 25)),
                f"12-{self.rng.integers(100, 999)}-{self.rng.integers(100, 999)}",
                float(self.rng.uniform(-500.0, 9500.0)),
                str(self.rng.choice(mktsegments))
            ))
        cur.executemany("INSERT INTO customer VALUES (?, ?, ?, ?, ?, ?, ?)", cust_rows)

        # 2. Suppliers
        supp_rows = []
        for i in range(1, n_supp + 1):
            supp_rows.append((
                i, f"Supplier#{i:08d}", f"SuppAddress_{i}", int(self.rng.integers(0, 25)),
                f"24-{self.rng.integers(100, 999)}-{self.rng.integers(100, 999)}",
                float(self.rng.uniform(-200.0, 8000.0))
            ))
        cur.executemany("INSERT INTO supplier VALUES (?, ?, ?, ?, ?, ?)", supp_rows)

        # 3. Parts
        part_rows = []
        for i in range(1, n_part + 1):
            part_rows.append((
                i, f"Part_Name_{i}", f"MFGR#{self.rng.integers(1, 6)}", f"Brand#{self.rng.integers(1, 6)}",
                "PROMO BURNISHED BRASS", int(self.rng.integers(1, 50)), float(self.rng.uniform(900.0, 2000.0))
            ))
        cur.executemany("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?)", part_rows)

        # 4. Orders
        order_rows = []
        for i in range(1, n_orders + 1):
            yr = self.rng.integers(1992, 1998)
            mo = self.rng.integers(1, 13)
            dy = self.rng.integers(1, 28)
            date_str = f"{yr:04d}-{mo:02d}-{dy:02d}"
            order_rows.append((
                i, int(self.rng.integers(1, n_cust + 1)), str(self.rng.choice(statuses)),
                float(self.rng.uniform(1000.0, 50000.0)), date_str,
                str(self.rng.choice(priorities)), f"Clerk#{self.rng.integers(1, 100):08d}", 0
            ))
        cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", order_rows)

        # 5. LineItem
        line_rows = []
        for i in range(1, n_lines + 1):
            yr = self.rng.integers(1992, 1998)
            mo = self.rng.integers(1, 13)
            dy = self.rng.integers(1, 28)
            date_str = f"{yr:04d}-{mo:02d}-{dy:02d}"
            line_rows.append((
                int(self.rng.integers(1, n_orders + 1)), int(self.rng.integers(1, n_part + 1)),
                int(self.rng.integers(1, n_supp + 1)), int(self.rng.integers(1, 5)),
                float(self.rng.uniform(1.0, 50.0)), float(self.rng.uniform(100.0, 5000.0)),
                float(self.rng.uniform(0.01, 0.10)), float(self.rng.uniform(0.01, 0.08)),
                'N', 'O', date_str, date_str, date_str
            ))
        cur.executemany("INSERT INTO lineitem VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", line_rows)
        self.conn.commit()

    def get_queries(self) -> Dict[str, str]:
        """Representative analytical TPC-H query workload."""
        return {
            "Q1_Lineitem_Aggregation": """
                SELECT l_returnflag, l_linestatus,
                       sum(l_quantity) as sum_qty,
                       sum(l_extendedprice) as sum_base_price,
                       avg(l_discount) as avg_disc
                FROM lineitem
                WHERE l_shipdate <= '1995-12-01'
                GROUP BY l_returnflag, l_linestatus;
            """,
            "Q2_Customer_Orders_Join": """
                SELECT c_custkey, c_name, count(o_orderkey) as order_count, sum(o_totalprice) as total_spent
                FROM customer
                JOIN orders ON c_custkey = o_custkey
                WHERE c_mktsegment = 'BUILDING' AND o_orderdate >= '1994-01-01'
                GROUP BY c_custkey, c_name
                ORDER BY total_spent DESC
                LIMIT 10;
            """,
            "Q3_Multi_Table_Join": """
                SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue,
                       o_orderdate, o_shippriority
                FROM customer, orders, lineitem
                WHERE c_mktsegment = 'AUTOMOBILE'
                  AND c_custkey = o_custkey
                  AND l_orderkey = o_orderkey
                  AND o_orderdate < '1995-03-15'
                  AND l_shipdate > '1995-03-15'
                GROUP BY l_orderkey, o_orderdate, o_shippriority
                ORDER BY revenue DESC
                LIMIT 10;
            """
        }
