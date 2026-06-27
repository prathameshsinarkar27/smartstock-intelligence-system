-- schema.sql
--
-- =====================================================
-- SmartStock Intelligence Platform
-- Database Initialization
-- =====================================================

-- Create the SmartStock database manually if it does not already exist.
-- Example:
--   CREATE DATABASE smartstock;

-- After creating the database, connect to it:
--   \c smartstock

-- Enable required PostgreSQL extensions.
-- pgcrypto will be used in later phases for UUID generation.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
