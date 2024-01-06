from .base_paras import num_anchor_state, num_anchor_per_step, num_inp_1, num_inp_2

base_size = 64
multi_head_size = 4
sizes = {
    'sequence_length_inp_1': num_inp_1,
    'sequence_length_inp_2': num_inp_2,
    'sequence_length_middle': base_size,
    'sequence_length_oup': num_anchor_per_step,
    'multi_head_size': multi_head_size,
    'state_size': num_anchor_state
}
