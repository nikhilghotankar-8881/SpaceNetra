-- SpaceNetra PostGIS Database Initialization Script
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create Scenes Table
CREATE TABLE IF NOT EXISTS scenes (
    id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    acquisition_date TIMESTAMP,
    cloud_cover_percentage FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Tiles Table
CREATE TABLE IF NOT EXISTS tiles (
    id VARCHAR(64) PRIMARY KEY,
    scene_id VARCHAR(64) REFERENCES scenes(id) ON DELETE CASCADE,
    tile_index_x INT NOT NULL,
    tile_index_y INT NOT NULL,
    bounding_box VARCHAR(128)
);

-- Create Change Events Table with PostGIS spatial geometry
CREATE TABLE IF NOT EXISTS change_events (
    event_id VARCHAR(64) PRIMARY KEY,
    scene_t1_id VARCHAR(64) REFERENCES scenes(id),
    scene_t2_id VARCHAR(64) REFERENCES scenes(id),
    centroid_lat FLOAT NOT NULL,
    centroid_lon FLOAT NOT NULL,
    area_sq_meters FLOAT NOT NULL,
    change_type VARCHAR(64) DEFAULT 'BUILDING_CONSTRUCTION',
    confidence_score FLOAT NOT NULL,
    geom GEOMETRY(Polygon, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Analyst Feedback Table
CREATE TABLE IF NOT EXISTS analyst_feedback (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(64) REFERENCES change_events(event_id) ON DELETE CASCADE,
    analyst_id VARCHAR(64) NOT NULL,
    decision VARCHAR(32) NOT NULL, -- ACCEPT, REJECT, REVIEW
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Prediction Provenance Table
CREATE TABLE IF NOT EXISTS model_provenance (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(64) REFERENCES change_events(event_id) ON DELETE CASCADE,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    weights_sha256 VARCHAR(64) NOT NULL,
    input_t1_sha256 VARCHAR(64) NOT NULL,
    input_t2_sha256 VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Spatial Index on Change Events
CREATE INDEX IF NOT EXISTS idx_change_events_geom ON change_events USING GIST(geom);
