#!/bin/bash
set -e
MODELS=("multi-qa-mpnet-base-cos-v1" "multi-qa-MiniLM-L6-cos-v1" "multi-qa-distilbert-cos-v1")

for MODEL in "${MODELS[@]}"
do
  echo ">>> PROVISIONING MODEL: $MODEL"
  curl -s -X POST http://127.0.0.1:5000/api/settings/embedding_model -H "Content-Type: application/json" -d "{\"model\": \"$MODEL\"}"
  
  echo ">>> RESETTING INGESTION TRACKER"
  rm -f ingested_files.json
  
  echo ">>> STARTING INGESTION (QDRANT)"
  ./venv/bin/python3 test_embedding_qdrant.py
  
  echo ">>> EXECUTING AUDIT METRICS"
  ./venv/bin/python3 run_metrics.py
  
  echo ">>> ARCHIVING REPORT FOR $MODEL"
  cp evaluation_results.json "report_${MODEL}.json"
done

echo ">>> ALL EVALUATIONS COMPLETE"
