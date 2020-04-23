#!/bin/bash
source activate ML
EI_NAME="py-ai" APP_NAME="py-ai-app" DC="dc" node-red -p 1881 -u py-ai flows.json
