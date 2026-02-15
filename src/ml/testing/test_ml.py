"""
Model Testing Module
Unit and integration tests for the ML pipeline
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "training"))

from train import (
    m3, clamp, compute_priority, connect_db, 
    load_slots_df, load_items_df, oracle_best_slot
)


class TestHelpers:
    """Test helper functions"""
    
    def test_m3_volume(self):
        """Test volume calculation"""
        assert m3(1, 1, 1) == 1.0
        assert m3(2, 2, 2) == 8.0
        
    def test_clamp(self):
        """Test clamp function"""
        assert clamp(0.5, 0, 1) == 0.5
        assert clamp(-1, 0, 1) == 0
        assert clamp(2, 0, 1) == 1


class TestDatabase:
    """Test database operations"""
    
    def test_connect_db(self):
        """Test database connection"""
        conn = connect_db()
        assert conn is not None
        conn.close()
    
    def test_load_slots(self):
        """Test loading slots"""
        conn = connect_db()
        slots_df = load_slots_df(conn)
        assert len(slots_df) > 0
        assert "slot_id" in slots_df.columns
        conn.close()
    
    def test_load_items(self):
        """Test loading items"""
        conn = connect_db()
        items_df = load_items_df(conn)
        assert len(items_df) > 0
        assert "item_id" in items_df.columns
        conn.close()


class TestOracle:
    """Test oracle function"""
    
    def test_oracle_finds_slot(self):
        """Test that oracle finds valid slots"""
        conn = connect_db()
        slots_df = load_slots_df(conn)
        items_df = load_items_df(conn)
        
        for _ in range(5):
            item = items_df.sample(1).iloc[0]
            slot_id, cost = oracle_best_slot(item, slots_df)
            
            if slot_id:
                # Verify slot is feasible
                slot = slots_df[slots_df["slot_id"] == slot_id].iloc[0]
                assert slot["Lm"] >= item["Lm"]
                assert slot["Wm"] >= item["Wm"]
                assert slot["Hm"] >= item["Hm"]
                assert slot["max_weight"] >= item["weight"]
        
        conn.close()


class TestPriority:
    """Test priority computation"""
    
    def test_priority_in_range(self):
        """Test that priority is in [0, 1]"""
        conn = connect_db()
        items_df = load_items_df(conn)
        
        for _, item in items_df.sample(10).iterrows():
            p = compute_priority(item)
            assert 0 <= p <= 1
        
        conn.close()


def run_tests():
    """Run all tests"""
    print("🧪 Running tests...\n")
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
