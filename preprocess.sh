./blender-4.3.2-linux-x64/blender -b -P utils/blender_preprocess.py -- \
   --input_path ./examples/cycle_enhance/example/source.glb\
   --output_dir ./examples/cycle_enhance/example \
   --radius 1.0 \
   --use_emission_shader \
   --random_camera \
   --num_renders 100 \
   --samples 256 \
   --axis_forward='Z'