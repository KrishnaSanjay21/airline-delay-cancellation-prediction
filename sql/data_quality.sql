-- Execute with DuckDB against data/processed/model_features.parquet.
CREATE OR REPLACE VIEW model_features AS
SELECT * FROM read_parquet('data/processed/model_features.parquet');

-- Natural-key collisions require source investigation before modeling.
SELECT flight_id, count(*) AS row_count
FROM model_features
GROUP BY flight_id
HAVING count(*) > 1;

-- Targets must remain binary and cancelled flights have no delay target.
SELECT
    count(*) FILTER (WHERE cancel_target NOT IN (0, 1)) AS invalid_cancellation_targets,
    count(*) FILTER (WHERE delay_target NOT IN (0, 1) AND delay_target IS NOT NULL)
        AS invalid_delay_targets,
    count(*) FILTER (WHERE cancel_target = 1 AND delay_target IS NOT NULL)
        AS cancelled_with_delay_target
FROM model_features;

-- Monitor feature availability by airport instead of hiding missing weather.
SELECT
    Origin,
    count(*) AS flights,
    avg(CASE WHEN origin_tmax IS NULL THEN 1 ELSE 0 END) AS missing_origin_weather_rate
FROM model_features
GROUP BY Origin
ORDER BY missing_origin_weather_rate DESC, flights DESC;

