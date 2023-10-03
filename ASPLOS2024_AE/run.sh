# training with 5 epochs, parameters are default
python tutorial_01_raw.py --depth 5 --amp-factor 2 --lr 0.2 --pad 50 --sys-size 200 --pixel-size 3.6e-5 --distance 0.3 --epochs 50

# inference with trained model
python tutorial_01_raw.py --evaluation True --whether-load-model True --start-epoch 50 --model-save-path ./save_model_raw/

# propagation visualization
python tutorial_01_raw.py --whether-load-model True --start-epoch 50 --model-save-path ./save_model_raw/ --prop_vis True

# phase visualization
python tutorial_01_raw.py --whether-load-model True --start-epoch 50 --model-save-path ./save_model_raw/ --phase_vis True

# codesign
python tutorial_01_codesign.py --lr 10 --amp-factor 3 --precision 16 --epochs 50

# inference 
python tutorial_01_codesign.py --lr 10 --amp-factor 3 --precision 16 --evaluation True --whether-load-model True --start-epoch 50 --model-save-path ./save_model_codesign/

# prop
python tutorial_01_codesign.py --lr 10 --amp-factor 3 --precision 16 --whether-load-model True --start-epoch 50 --model-save-path ./save_model_codesign/ --prop_vis True

# phase
python tutorial_01_codesign.py --lr 10 --amp-factor 3 --precision 16 --whether-load-model True --start-epoch 50 --model-save-path ./save_model_codesign/ --phase_vis True

