#!/bin/bash
exec > evaluation.log 2>&1
python dev_inference.py --mode evaluate --model_path saved_models/checkpoint-1000 --dev_data_path dev_data/ --output_filepath results/results_checkpoint-1000_all.json --use_adapter
python dev_inference.py --mode evaluate --model_path saved_models/gemma-checkpoint-1400 --dev_data_path dev_data/ --output_filepath results/results_gemma-checkpoint-1400_all.json --use_adapter
python dev_inference.py --mode evaluate --model_path saved_models/mistral_raid_detector_adapter --dev_data_path dev_data/ --output_filepath results/results_mistral_raid_detector_adapter_all.json --use_adapter 