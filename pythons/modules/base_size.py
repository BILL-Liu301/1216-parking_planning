from .base_paras import num_anchor_state, num_anchors_pre

base_size = 64
multi_head_size = 6
sizes = {
    "sequence_length": num_anchors_pre,
    "multi_head_size": multi_head_size,
    "encoder_input_size": num_anchor_state,
    "encoder_middle_size": base_size,
    "encoder_output_size": num_anchor_state,
    "decoder_input_size": num_anchor_state,
    "decoder_middle_size": base_size,
    "decoder_output_size": num_anchor_state
}
