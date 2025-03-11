import tf_keras as tfk
import keras
from transformers import TFBertModel, BertTokenizer

@keras.saving.register_keras_serializable(package="app")
class ResumeAnalyzerModel(keras.Model):
    def __init__(self, transformer, num_labels, **kwargs):
        super(ResumeAnalyzerModel, self).__init__()
        self.transformer = transformer
        self.num_labels = num_labels
        self.token_classifier = keras.layers.Dense(num_labels, activation='softmax')
        self.regressor = keras.layers.Dense(1, activation='linear')
    
    def get_config(self):
        config = super().get_config()
        # config.update({
        #     "transformer_name": self.transformer.name_or_path if hasattr(self.transformer, "name_or_path") else None,
        #     "num_labels": self.num_labels
        # })
        config.update({
            "transformer_config": {
                "name": "bert-base-uncased"  # Store model name instead of object
            },
            "num_labels": self.num_labels
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        # transformer_name = config.pop("bert-base-uncased")
        # transformer = TFBertModel.from_pretrained(transformer_name)
        transformer_config = config.pop("transformer_config")
        transformer = TFBertModel.from_pretrained(transformer_config["name"])
        num_labels = config.pop("num_labels")
        return cls(transformer=transformer, num_labels=num_labels, **config)
        
           
    def call(self, inputs, training=False):
        outputs = self.transformer(**inputs)
        sequence_output = outputs.last_hidden_state
        token_logits = self.token_classifier(sequence_output)
        cls_output = sequence_output[:, 0, :]
        overall_score = self.regressor(cls_output)
        return token_logits, overall_score

def get_model(num_labels: int = 5):
    """
    Load the BERT model and tokenizer, then instantiate the ResumeAnalyzerModel.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transformer = TFBertModel.from_pretrained('bert-base-uncased')
    model = ResumeAnalyzerModel(transformer, num_labels)
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, 
                  loss=[keras.losses.SparseCategoricalCrossentropy(), keras.losses.MeanSquaredError()])
    return model, tokenizer