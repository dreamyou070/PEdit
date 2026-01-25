import torch
from torch.nn import functional as F
from diffusers.models.embeddings import apply_rotary_emb

class Controller():

    def __init__(self):

        self.valid_len = 0
        self.tca_dict = {}
        self.ica_dict = {}
        self.txt_alpha_dict = {}
        self.img_alpha_dict = {}
        self.state = ""
        self.embedding_dictionary = {}

    def set_alpha(self, attn_idx, txt_alpha, img_alpha):

        self.txt_alpha_dict[attn_idx] = txt_alpha
        self.img_alpha_dict[attn_idx] = img_alpha

    def reset(self):

        self.embedding_dictionary = {}
        self.tca_dict = {}
        self.ica_dict = {}


def set_alphas_for_case(transformer, model_name, controller, case, config_dict):

    if model_name == 'Kontext' :

        def ca_forward_kontext(attn_module, layer_idx, controller):

            def forward(hidden_states: torch.FloatTensor,
                        encoder_hidden_states: torch.FloatTensor = None,
                        attention_mask=None,
                        image_rotary_emb=None,
                        **kwargs) -> torch.FloatTensor:

                attn = attn_module
                is_cross = encoder_hidden_states is not None
                txt_alpha = controller.txt_alpha_dict[str(layer_idx)]
                img_alpha = controller.img_alpha_dict[str(layer_idx)]

                controller.embedding_dictionary[str(layer_idx)] = {}

                if is_cross:
                    encoder_hidden_states *= txt_alpha

                    controller.embedding_dictionary[str(layer_idx)]['txt'] = encoder_hidden_states

                    total_len = hidden_states.size(1)
                    source_len = total_len // 2
                    image_hidden_states = hidden_states[:, -source_len:] * img_alpha

                    controller.embedding_dictionary[str(layer_idx)]['img'] = image_hidden_states

                    hidden_states[:, -source_len:] = image_hidden_states
                else:
                    hidden_states[:, :512] *= txt_alpha
                    controller.embedding_dictionary[str(layer_idx)]['txt'] = hidden_states[:, :512]
                    total_len = hidden_states.size(1)
                    source_len = (total_len - 512) // 2
                    hidden_states[:, -source_len:] *= img_alpha
                    controller.embedding_dictionary[str(layer_idx)]['img'] = hidden_states[:, -source_len:]
                batch_size, _, _ = hidden_states.shape if not is_cross else encoder_hidden_states.shape
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q: query = attn.norm_q(query)
                if attn.norm_k: key = attn.norm_k(key)

                if is_cross:
                    def proj(tensor, proj_fn, norm_fn):
                        out = proj_fn(tensor).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        return norm_fn(out) if norm_fn else out

                    query = torch.cat([proj(encoder_hidden_states, attn.add_q_proj, attn.norm_added_q), query], dim=2)
                    key = torch.cat([proj(encoder_hidden_states, attn.add_k_proj, attn.norm_added_k), key], dim=2)
                    value = torch.cat([proj(encoder_hidden_states, attn.add_v_proj, None), value], dim=2)

                if image_rotary_emb is not None:
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

                # ---------------------------------------------------------------------------------------------------------
                # calculate attention score
                # ---------------------------------------------------------------------------------------------------------

                with torch.no_grad():

                    def get_attention(query, key):
                        B, H, _, D = query.shape
                        q = query[:, :, 512:512 + 4096, :]
                        attn_weight = torch.matmul(q, key.transpose(-2, -1))
                        txt_len = int(controller.valid_len)
                        t = attn_weight[:,:,:,:txt_len]
                        i = attn_weight[:,:,:,-4096:]
                        attn_weight = torch.softmax(torch.cat([t,i], dim = -1), dim=-1)
                        TCA_full = attn_weight[:, :, :, :txt_len]
                        ICA_full = attn_weight[:, :, :, -4096:]
                        diag_ICA = torch.diagonal(ICA_full, dim1=-2, dim2=-1)  # [B, H, 4096]
                        TCA_mean_over_text = TCA_full.sum(dim=-1)
                        TCA_spatial = TCA_mean_over_text.sum(dim=0).sum(dim=0).float()  # [4096]
                        TCA_value = TCA_spatial.sum()
                        ICA_spatial = diag_ICA.sum(dim=0).sum(dim=0).float()  # [4096]
                        ICA_value = ICA_spatial.sum()
                        return TCA_value, ICA_value
                    TCA_value, ICA_value = get_attention(query, key)
                    controller.tca_dict[str(layer_idx)] = TCA_value
                    controller.ica_dict[str(layer_idx)] = ICA_value

                hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
                if is_cross:
                    encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                        [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                        dim=1)
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                    return hidden_states, encoder_hidden_states
                else:
                    return hidden_states

            return forward

        attention_id = 0


        for nm, module in transformer.named_modules():
            if module.__class__.__name__ in ('FluxAttention', 'Attention'):

                txt_alpha = torch.as_tensor(config_dict[f'{str(attention_id)}_txt_alpha'])
                img_alpha = torch.as_tensor(config_dict[f'{str(attention_id)}_img_alpha'])
                if case == "case1":
                    txt_alpha = torch.clamp(txt_alpha, 0.2, 1.0)
                    img_alpha = torch.clamp(img_alpha, 0.2, 1.0)
                controller.set_alpha(str(attention_id), txt_alpha=txt_alpha, img_alpha= img_alpha)
                module.forward = ca_forward_kontext(module, attention_id, controller)
                attention_id += 1
    else :

        from diffusers.models.attention_dispatch import dispatch_attention_fn

        def apply_rotary_emb_qwen(
                x: torch.Tensor,
                freqs_cis,
                use_real: bool = True,
                use_real_unbind_dim: int = -1, ):

            if use_real:
                cos, sin = freqs_cis  # [S, D]
                cos = cos[None, None]
                sin = sin[None, None]
                cos, sin = cos.to(x.device), sin.to(x.device)

                if use_real_unbind_dim == -1:
                    # Used for flux, cogvideox, hunyuan-dit
                    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
                    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
                elif use_real_unbind_dim == -2:
                    # Used for Stable Audio, OmniGen, CogView4 and Cosmos
                    x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
                    x_rotated = torch.cat([-x_imag, x_real], dim=-1)
                else:
                    raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

                out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

                return out
            else:
                x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(1)
                x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

                return x_out.type_as(x)

        def ca_forward(attn, layer_idx, controller):

            def forward(hidden_states: torch.FloatTensor,  # batch,8192,3072
                        encoder_hidden_states: torch.FloatTensor = None,  # batch, text_len, 3072
                        encoder_hidden_states_mask: torch.FloatTensor = None,
                        attention_mask=None,
                        image_rotary_emb=None,
                        ) -> torch.FloatTensor:

                if encoder_hidden_states is None:
                    raise ValueError(
                        "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

                seq_txt = encoder_hidden_states.shape[1]

                txt_alpha = controller.txt_alpha_dict[str(layer_idx)]
                img_alpha = controller.img_alpha_dict[str(layer_idx)]

                img_query = attn.to_q(hidden_states)
                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)

                encoder_hidden_states = encoder_hidden_states * txt_alpha
                txt_query = attn.add_q_proj(encoder_hidden_states)
                txt_key = attn.add_k_proj(encoder_hidden_states)
                txt_value = attn.add_v_proj(encoder_hidden_states)
                img_query = img_query.unflatten(-1, (attn.heads, -1))  # let it be 4th dim
                img_key = img_key.unflatten(-1, (attn.heads, -1))
                img_value = img_value.unflatten(-1, (attn.heads, -1))
                txt_query = txt_query.unflatten(-1, (attn.heads, -1))
                txt_key = txt_key.unflatten(-1, (attn.heads, -1))
                txt_value = txt_value.unflatten(-1, (attn.heads, -1))
                # Apply QK normalization
                if attn.norm_q is not None:
                    img_query = attn.norm_q(img_query)
                if attn.norm_k is not None:
                    img_key = attn.norm_k(img_key)

                if attn.norm_added_q is not None:
                    txt_query = attn.norm_added_q(txt_query)
                if attn.norm_added_k is not None:
                    txt_key = attn.norm_added_k(txt_key)

                # Apply RoPE
                if image_rotary_emb is not None:
                    img_freqs, txt_freqs = image_rotary_emb
                    img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                    img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

                joint_query = torch.cat([txt_query, img_query], dim=1)  #
                joint_key = torch.cat([txt_key, img_key], dim=1)
                joint_value = torch.cat([txt_value, img_value], dim=1)
                with torch.no_grad():

                    def get_attention(query, key):
                        B, H, _, D = query.shape
                        query = query.permute(0, 2, 1, 3).mean(dim=0).mean(dim=0)
                        key = key.permute(0, 2, 1, 3).mean(dim=0).mean(dim=0)
                        q = query[-4096 * 2: -4096, :]
                        txt_len = int(controller.valid_len)
                        k1 = key[:txt_len, :]
                        k2 = key[-4096:, :]
                        key = torch.cat([k1, k2], dim=-2)
                        attn_weight = torch.matmul(q, key.transpose(-2, -1))
                        attn_weight = torch.softmax(attn_weight, dim=-1)

                        # 3) 분리
                        TCA_full = attn_weight[:, :txt_len]
                        TCA_mean_over_text = TCA_full.sum()
                        TCA_spatial = TCA_mean_over_text
                        TCA_value = TCA_spatial

                        ICA_full = attn_weight[:, -4096:]
                        diag_ICA = torch.diagonal(ICA_full).sum()  # [B, H, 4096]
                        ICA_spatial = diag_ICA
                        ICA_value = ICA_spatial.sum()
                        return TCA_value, ICA_value

                    TCA_value, ICA_value = get_attention(joint_query, joint_key)

                if layer_idx not in controller.tca_dict:
                    controller.tca_dict[layer_idx] = []
                    controller.ica_dict[layer_idx] = []

                controller.tca_dict[layer_idx] = TCA_value
                controller.ica_dict[layer_idx] = ICA_value

                joint_hidden_states = dispatch_attention_fn(
                    joint_query,
                    joint_key,
                    joint_value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=attn.processor._attention_backend,
                )
                joint_hidden_states = joint_hidden_states.flatten(2, 3)
                joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
                # 이게 이미 attention 을 지나온 것이라고 생각을 할 수 있음

                txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part

                img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part
                # when zeroiung image zttenion
                src_len = 4096
                img_attn_output[:, -src_len, :] = img_attn_output[:, -src_len, :] * img_alpha

                # ------------------------------------------------------------------------- #
                # [2] Value : Apply output projections
                img_attn_output = attn.to_out[0](img_attn_output)
                if len(attn.to_out) > 1:
                    img_attn_output = attn.to_out[1](img_attn_output)  # dropout
                txt_attn_output = attn.to_add_out(txt_attn_output)
                return img_attn_output, txt_attn_output

            return forward

        attention_id = 0

        for nm, module in transformer.named_modules():

            if module.__class__.__name__ in ('Attention'):

                txt_alpha = torch.as_tensor(config_dict[f'{str(attention_id)}_txt_alpha'])
                img_alpha = torch.as_tensor(config_dict[f'{str(attention_id)}_img_alpha'])
                if case == "case1":
                    txt_alpha = torch.clamp(txt_alpha, 0.2, 1.2)
                    img_alpha = torch.clamp(img_alpha, 0.2, 1.2)
                controller.set_alpha(str(attention_id), txt_alpha=txt_alpha, img_alpha=img_alpha)
                module.forward = ca_forward(module, attention_id, controller)
                attention_id += 1


def set_alpha_dict(model, transformer, case,
                   case2_s, case3_s,
                   SIB=None,
                   STB=None,
                   data=None):

    config_dict = {}
    training_models = []
    if model == 'Kontext' :
        BASE = 1.0  # 기준 1.0

        if data == 'hq' :
            #SIB = [1, 0, 39, 2, 13, 43, 36, 47, 51, 54, 19, 41,26,16,22,49,20,55,46] #,,12,3,7,10,23,14,8,53,35,9,52,40,21]
            #STB = [1, 3, 28, 11, 18, 31, 12, 25, 35, 22, 5, 19,41,26,16,22,49,20,26,46,47,14,7,16,39,23 ,51,54,49,0,4,40,44,8,17] # 21] #,36,55] #,42,30,56,6,2,10,33,34,15
            #EIB = [31, 37, 29, ] #32, 17, ] # 18, 38]#, 50, 42, 34 ] # , 24, 27, 44]
            #ETB = [27, 45, 37, 52, 24, 38, 48, 50, 32, 43, 9]
            SIB = [1, 0, 39, 2, 13, 43, 36, 47, 51] #,,12,3,7,10,23,14,8,53,35,9,52,40,21]
            STB = [1, 3, 28, 11, 18, 31, 12, 25, 35, 22, 5, 19,41,26,16,22,49,20,26,46,47,14,7,16,39,23,51,54,49,0,4,40,44,8,17] # 21] #,36,55] #,42,30,56,6,2,10,33,34,15
            #EIB = [31, 37, 29, ] #32, 17, ] # 18, 38]#, 50, 42, 34 ] # , 24, 27, 44]
            #ETB = [27, 45, 37, 52, 24, 38, 48, 50, 32, 43, 9]
            EIB = [31]
            ETB = [27, 45]

        if data == 'emu' :
            # SIB = [1, 0, 39, 2, 13, 43, 36, 47, 51, 54, 19, 41,26,16,22,49,20,55,46] #,,12,3,7,10,23,14,8,53,35,9,52,40,21]
            # STB = [1, 3, 28, 11, 18, 31, 12, 25, 35, 22, 5, 19,41,26,16,22,49,20,26,46,47,14,7,16,39,23 ,51,54,49,0,4,40,44,8,17] # 21] #,36,55] #,42,30,56,6,2,10,33,34,15
            # EIB = [31, 37, 29, ] #32, 17, ] # 18, 38]#, 50, 42, 34 ] # , 24, 27, 44]
            # ETB = [27, 45, 37, 52, 24, 38, 48, 50, 32, 43, 9]
            SIB = [1, 0, 39, 2, 13, 43, 36, 47, 51]  # ,,12,3,7,10,23,14,8,53,35,9,52,40,21]
            STB = [1, 3, 28, 11, 18, 31, 12, 25, 35, 22, 5, 19, 41, 26, 16, 22, 49, 20, 26, 46, 47, 14, 7, 16, 39, 23,51, 54, 49, 0, 4, 40, 44, 8, 17]  # 21] #,36,55] #,42,30,56,6,2,10,33,34,15
            EIB = [31, 37, 29, ]  # 32, 17, ] # 18, 38]#, 50, 42, 34 ] # , 24, 27, 44]
            ETB = [27, 45, 37, 52, 24, 38, 48, 50, 32, 43, 9]

        if case == 'case2':
            attn_id = 0
            for nm, module in transformer.named_modules():
                if module.__class__.__name__ in ('FluxAttention', 'Attention'):
                    img_alpha = 1
                    txt_alpha = 1
                    if attn_id in SIB:  # if down img
                        img_alpha = BASE - 0.1 * case2_s
                        training_models.append(f'{str(attn_id)}_img_alpha')
                    if attn_id in STB:
                        txt_alpha = BASE - 0.1 * case2_s
                        training_models.append(f'{str(attn_id)}_txt_alpha')
                    config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                    config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                    attn_id += 1
        if case == 'case3':
            #if use_case2 :
            if data == 'emu' :
                SIB = [1, 0, 39, 2, 13, 43, 36, 47 ]
                STB = [1, 3, 28, 11, 18, 31, 12, 25, 35, 22, 5, 19, 41, 26, 16, 22,49, 20, 26, 46, 47, 14, 7, 16, 39]
                attn_id = 0
                for nm, module in transformer.named_modules():
                    if module.__class__.__name__ in ('FluxAttention', 'Attention'):
                        img_alpha = 1
                        txt_alpha = 1
                        if attn_id in SIB:  # if down img
                            img_alpha = BASE - 0.1 * case3_s
                            training_models.append(f'{str(attn_id)}_img_alpha')
                        if attn_id in STB:
                            txt_alpha = BASE - 0.1 * case3_s
                            training_models.append(f'{str(attn_id)}_txt_alpha')
                        config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                        config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                        attn_id += 1
            else :
                attn_id = 0
                for nm, module in transformer.named_modules():
                    if module.__class__.__name__ in ('FluxAttention', 'Attention'):
                        " Editing "
                        img_alpha = 1
                        txt_alpha = 1
                        if attn_id in EIB:  # if down img
                            img_alpha = BASE - 0.1 * case3_s
                            training_models.append(f'{str(attn_id)}_img_alpha')
                        if attn_id in ETB:
                            txt_alpha = BASE - 0.1 * case3_s
                            training_models.append(f'{str(attn_id)}_txt_alpha')
                        config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                        config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                        attn_id += 1
        if case == 'case1':
            attn_id = 0
            for nm, module in transformer.named_modules():
                if module.__class__.__name__ in ('FluxAttention', 'Attention'):
                    " Structure Preserving "
                    img_alpha = 1
                    txt_alpha = 1
                    config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                    config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                    attn_id += 1

    if model == 'Qwen' :

        SIB = [19]
        STB = [4, 19, 31, 44, 50]
        EIB = [26, 57]
        ETB = [3, 56]

        BASE = 1.0  # 기준 1.0
        config_dict = {}
        if case == 'case3':
            attn_id = 0
            for nm, module in transformer.named_modules():
                if module.__class__.__name__ in ('Attention'):
                    " Structure Preserving "
                    img_alpha = 1
                    txt_alpha = 1
                    if attn_id in EIB:  # if down img
                        img_alpha = BASE - 0.1 * case3_s
                        training_models.append(f'{str(attn_id)}_img_alpha')
                    if attn_id in ETB:
                        print(f' [CASE2] Reducing alpha')
                        txt_alpha = BASE - 0.1 * case3_s
                        training_models.append(f'{str(attn_id)}_txt_alpha')
                    config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                    config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                    attn_id += 1

        if case == 'case2':
            attn_id = 0
            for nm, module in transformer.named_modules():
                if module.__class__.__name__ in ('FluxAttention', 'Attention'):
                    img_alpha = 1
                    txt_alpha = 1
                    if attn_id in SIB:  # if down img
                        img_alpha = BASE - 0.1 * case2_s
                        training_models.append(f'{str(attn_id)}_img_alpha')
                    if attn_id in STB:
                        txt_alpha = BASE - 0.1 * case2_s
                        training_models.append(f'{str(attn_id)}_txt_alpha')
                    config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                    config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                    attn_id += 1

        if case == 'case1':
            attn_id = 0
            for nm, module in transformer.named_modules():
                if module.__class__.__name__ in ('FluxAttention', 'Attention'):
                    " Structure Preserving "
                    img_alpha = 1
                    txt_alpha = 1
                    config_dict[f'{str(attn_id)}_txt_alpha'] = txt_alpha
                    config_dict[f'{str(attn_id)}_img_alpha'] = img_alpha
                    attn_id += 1

    return config_dict, training_models