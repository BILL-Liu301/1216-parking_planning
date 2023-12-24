from .base_paras import num_anchor_state, num_anchor_per_step, num_anchor_inp

base_size = 64
multi_head_size = 4
sizes = {
    "sequence_length_inp": num_anchor_inp,
    "sequence_length_oup": num_anchor_per_step,
    "multi_head_size": multi_head_size,
    "encoder_input_size": num_anchor_state,
    "encoder_middle_size": base_size,
    "encoder_output_size": num_anchor_state,
    "decoder_input_size": num_anchor_state,
    "decoder_middle_size": base_size,
    "decoder_output_size": num_anchor_state
}
