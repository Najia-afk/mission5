from typing import Optional, List
import os
import pandas as pd
from sqlalchemy import create_engine, Engine, inspect
from pathlib import Path

class DatabaseConnection:
    """Database connection manager with caching capabilities."""
    
    def __init__(self, db_path: str) -> None:
        """Initialize database connection and cache directory."""
        self.db_path = db_path
        self._engine: Optional[Engine] = None
        script_dir = Path(__file__).resolve().parent
        self.cache_dir = script_dir.parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)

    def get_engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(f'sqlite:///{self.db_path}')
        return self._engine

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        sanitized_key = "".join(c for c in key if c.isalnum() or c in ('-', '_'))
        return self.cache_dir / f"{sanitized_key}.pkl"

    def _save_to_cache(self, df: pd.DataFrame, key: str) -> None:
        """Save DataFrame to cache."""
        try:
            cache_path = self._get_cache_path(key)
            df.to_pickle(str(cache_path))
        except Exception as e:
            print(f"Warning: Failed to save cache for {key}: {e}")

    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if exists."""
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                return pd.read_pickle(str(cache_path))
        except Exception as e:
            print(f"Warning: Failed to load cache for {key}: {e}")
        return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def get_table_names(self) -> List[str]:
        """Get all table names from database."""
        inspector = inspect(self.get_engine())
        return inspector.get_table_names()

    def read_table(self, table_name: str, use_cache: bool = True) -> pd.DataFrame:
        """Read table into DataFrame with optional caching."""
        if use_cache:
            cached_df = self._load_from_cache(table_name)
            if cached_df is not None:
                return cached_df
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.get_engine())
        if use_cache:
            self._save_to_cache(df, table_name)
        return df

    def execute_query(self, query: str, cache_key: Optional[str] = None) -> pd.DataFrame:
        """Execute SQL query with optional caching."""
        if cache_key:
            cached_df = self._load_from_cache(cache_key)
            if cached_df is not None:
                return cached_df
        
        df = pd.read_sql_query(query, self.get_engine())
        if cache_key:
            self._save_to_cache(df, cache_key)
        return df

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._engine:
            self._engine.dispose()


def main():
    """Main function to run the script as a standalone tool."""
    import argparse

    parser = argparse.ArgumentParser(description='SQLite Database Connector Tool')
    
    # Required database path argument
    parser.add_argument('-db_path', required=True, help='Path to the SQLite database file')
    
    # Optional command arguments - only one should be used at a time
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-list', action='store_true', help='List all tables in the database')
    group.add_argument('-query', help='Execute SQL query and display results')
    
    # Optional CSV output path
    parser.add_argument('-csv', help='Path to save results as CSV file')
    
    args = parser.parse_args()
    
    # Create database connection
    db = DatabaseConnection(args.db_path)
    
    try:
        if args.list:
            # List all tables in the database
            tables = db.get_table_names()
            if tables:
                print("Tables in database:")
                for table in tables:
                    print(f"  - {table}")
            else:
                print("No tables found in database.")
        
        elif args.query:
            # Execute the SQL query
            df = db.execute_query(args.query)
            
            # Save to CSV if requested
            if args.csv:
                df.to_csv(args.csv, index=False)
                print(f"Exported {len(df)} rows to {args.csv}")
                
            # Always display results in the console
            print(f"\nQuery results (showing first 10 rows):")
            print(df.head(10).to_string())
            print(f"\nTotal rows: {len(df)}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        db.__exit__(None, None, None)


if __name__ == "__main__":
    main()


#usage example:
#python .\src\scripts\sqlite_connector.py -db_path .\dataset\olist.db -query "SELECT s.seller_id, CAST(SUM(oi.price + oi.freight_value) AS INTEGER) as total_revenue FROM sellers s JOIN order_items oi ON s.seller_id = oi.seller_id JOIN orders o ON oi.order_id = o.order_id WHERE o.order_status = 'delivered' GROUP BY s.seller_id HAVING total_revenue > 100000 ORDER BY total_revenue DESC" -csv "query.csv"