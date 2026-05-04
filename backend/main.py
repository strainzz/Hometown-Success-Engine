import json
import logging
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

class CloudRunJSONFormatter(logging.Formatter):
def format(self, record: logging.LogRecord) -> str:
log_record = {
"severity": record.levelname,
"message": record.getMessage(),
"name": record.name,
"timestamp": self.formatTime(record, self.datefmt)
}
if record.exc_info:
log_record["exc_info"] = self.formatException(record.exc_info)
return json.dumps(log_record)

logger = logging.getLogger("hometown-success-engine")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CloudRunJSONFormatter())
logger.addHandler(handler)

app = FastAPI(title="Hometown Success Engine Backend")

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "*")

app.add_middleware(
CORSMiddleware,
allow_origins=[frontend_origin],
allow_credentials=True,
allow_methods=[""],
allow_headers=[""],
)

@app.get("/health")
def health_check() -> dict[str, str]:
logger.info("Health check endpoint called")
return {"status": "ok", "service": "hometown-success-engine-backend"}

if name == "main":
import uvicorn
port = int(os.environ.get("PORT", 8080))
uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)