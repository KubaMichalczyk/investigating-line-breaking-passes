%run -i read_data.py
%run -i detect_lines.py
%run -i compute_pitch_control.py
%run -i impute_intercepted_passes.py
%run -i define_line_breaking_pass.py
%run -i create_features_and_labels.py
%run -i compute_epv.py --train
