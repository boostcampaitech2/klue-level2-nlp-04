import torch
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel

from loss import LabelSmoothingLoss


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return self.linear(x)


class CustomModel(BertPreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, config=config)

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size)
        self.label_classifier = self.label_classifier = FCLayer(
            config.hidden_size * 5,
            config.num_labels,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e3_h = self.entity_average(sequence_output, e3_mask)
        e4_h = self.entity_average(sequence_output, e4_mask)

        # Concat -> fc_layer
        # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = nn.CrossEntropyLoss()
                loss_fct = LabelSmoothingLoss(smoothing=0.1)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class CustomModel2(BertPreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, config=config)

        self.num_labels = config.num_labels
        hidden_size = config.hidden_size

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size)
        self.label_classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, self.num_labels)
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                e1_mask=None, e2_mask=None, e3_mask=None, e4_mask=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        # e3_h = self.entity_average(sequence_output, e3_mask)
        # e4_h = self.entity_average(sequence_output, e4_mask)

        # Concat -> fc_layer
        # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        # concat_h = torch.cat([pooled_output, e1_h, e2_h, e3_h, e4_h], dim=-1)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                # loss_fct = LabelSmoothingLoss(smoothing=0.1)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
