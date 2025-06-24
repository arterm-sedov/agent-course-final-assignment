#!/usr/bin/env python3
"""
Clear the agent_course_reference table to fix duplicate data issues.
"""

import os
from dotenv import load_dotenv
from supabase.client import create_client

def clear_table():
    """Clear all records from the agent_course_reference table."""
    
    # Load environment variables
    load_dotenv()
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials in .env file")
        return False
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
        
        # Method 1: Try DELETE with WHERE clause
        print("🗑️  Attempting to clear table with DELETE...")
        try:
            response = supabase.table("agent_course_reference").delete().neq("id", 0).execute()
            print(f"✅ Successfully cleared {len(response.data) if response.data else 0} records")
            return True
        except Exception as e:
            print(f"⚠️  DELETE method failed: {e}")
        
        # Method 2: Try truncate function
        print("🗑️  Attempting to clear table with truncate function...")
        try:
            supabase.rpc('truncate_agent_course_reference').execute()
            print("✅ Successfully cleared table using truncate function")
            return True
        except Exception as e:
            print(f"⚠️  Truncate function failed: {e}")
        
        # Method 3: Try direct SQL
        print("🗑️  Attempting to clear table with direct SQL...")
        try:
            supabase.table("agent_course_reference").delete().execute()
            print("✅ Successfully cleared table using direct DELETE")
            return True
        except Exception as e:
            print(f"⚠️  Direct DELETE failed: {e}")
        
        print("❌ All clearing methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Clearing agent_course_reference table...")
    success = clear_table()
    if success:
        print("🎉 Table cleared successfully!")
    else:
        print("❌ Failed to clear table")
        print("\n💡 Manual SQL solution:")
        print("Run this SQL in your Supabase SQL editor:")
        print("DELETE FROM agent_course_reference;")
        print("-- OR --")
        print("TRUNCATE TABLE agent_course_reference RESTART IDENTITY;") 