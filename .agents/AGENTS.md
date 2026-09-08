# SpaceNetra Project Rules

### Real-Time Geospatial AI Engine First

1. **System Identity**: SpaceNetra is an enterprise real-time satellite change detection and geospatial intelligence engine.
2. **Local Dashboard Boundary**: The web application layer (`app/main.py`, `app/static/index.html`) functions as a local operational control panel and visualization interface. Do not simplify engine design into static web CRUD concepts.
3. **Stream & Real-Time Architecture**: All pipeline components (Sentinel ingestion, ChangeFormer inference, vector retrieval, event clustering, DB ORM) must retain low-latency, modular stream-ready design to support seamless deployment transitions to live streaming satellite feeds.
