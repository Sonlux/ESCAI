// Initialize ESCAI MongoDB Database
// This script runs when the MongoDB container starts for the first time

// Switch to the escai_mongo database
db = db.getSiblingDB("escai_mongo");

// Create collections with validation
db.createCollection("raw_logs", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["agent_id", "timestamp", "log_level", "message"],
      properties: {
        agent_id: { bsonType: "string" },
        timestamp: { bsonType: "date" },
        log_level: {
          bsonType: "string",
          enum: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        },
        message: { bsonType: "string" },
        metadata: { bsonType: "object" },
      },
    },
  },
});

db.createCollection("processed_events", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["event_id", "agent_id", "timestamp", "event_type"],
      properties: {
        event_id: { bsonType: "string" },
        agent_id: { bsonType: "string" },
        timestamp: { bsonType: "date" },
        event_type: { bsonType: "string" },
        data: { bsonType: "object" },
      },
    },
  },
});

db.createCollection("explanations", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["explanation_id", "agent_id", "timestamp", "explanation_text"],
      properties: {
        explanation_id: { bsonType: "string" },
        agent_id: { bsonType: "string" },
        timestamp: { bsonType: "date" },
        explanation_text: { bsonType: "string" },
        confidence: { bsonType: "double", minimum: 0, maximum: 1 },
      },
    },
  },
});

db.createCollection("configuration", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["config_key", "config_value"],
      properties: {
        config_key: { bsonType: "string" },
        config_value: { bsonType: "object" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" },
      },
    },
  },
});

// Create indexes for performance
db.raw_logs.createIndex({ agent_id: 1, timestamp: -1 });
db.raw_logs.createIndex({ timestamp: -1 });
db.raw_logs.createIndex({ log_level: 1 });

db.processed_events.createIndex({ agent_id: 1, timestamp: -1 });
db.processed_events.createIndex({ event_type: 1 });
db.processed_events.createIndex({ timestamp: -1 });

db.explanations.createIndex({ agent_id: 1, timestamp: -1 });
db.explanations.createIndex({ timestamp: -1 });

db.configuration.createIndex({ config_key: 1 }, { unique: true });

// Create TTL indexes for data retention (90 days default)
db.raw_logs.createIndex({ timestamp: 1 }, { expireAfterSeconds: 7776000 }); // 90 days
db.processed_events.createIndex(
  { timestamp: 1 },
  { expireAfterSeconds: 7776000 }
); // 90 days

print("ESCAI MongoDB initialization completed successfully");
