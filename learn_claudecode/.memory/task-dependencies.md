---
name: task-dependencies
description: Task dependencies for project: schema, endpoints, tests, docs
type: project
---

- setup database schema (no dependency)
- create API endpoints (depends on schema)
- write tests (depends on endpoints)
- write docs (depends on schema)
