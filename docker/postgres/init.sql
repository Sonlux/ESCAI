-- Initialize ESCAI PostgreSQL Database
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create indexes for performance
-- These will be created by Alembic migrations, but we ensure extensions are available

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE escai_db TO escai;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS escai;
GRANT ALL ON SCHEMA escai TO escai;

-- Set default search path
ALTER DATABASE escai_db SET search_path TO escai, public;