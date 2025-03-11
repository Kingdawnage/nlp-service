import tensorflow as tf
import tf_keras as tfk
import keras
from transformers import TFBertModel, BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer = TFBertModel.from_pretrained('bert-base-uncased')

@keras.saving.register_keras_serializable()
class ResumerAnalyzerModel(keras.Model):
    def __init__(self, transformer, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer
        self.num_labels = num_labels
        self.token_classifier = keras.layers.Dense(num_labels, activation='softmax')
        self.regressor = keras.layers.Dense(1, activation='linear')
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_labels": self.num_labels,
            "transformer": self.transformer
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Extract custom parameters
        transformer = config.pop("transformer")
        num_labels = config.pop("num_labels")
        # Create instance with remaining kwargs
        return cls(transformer=transformer, num_labels=num_labels, **config)
          
    def call(self, inputs, training=False):
        outputs = self.transformer(**inputs)
        sequence_output = outputs.last_hidden_state
        token_logits = self.token_classifier(sequence_output)
        cls_output = sequence_output[:, 0, :]
        overall_score = self.regressor(cls_output)
        return token_logits, overall_score
    
num_labels = 5

model = ResumerAnalyzerModel(transformer, num_labels)
optimizer = keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=[keras.losses.SparseCategoricalCrossentropy(), keras.losses.MeanSquaredError()])

sample_text = (
    "John Doe\n\n"
    "Summary\n"
    "Software engineer with experience in multiple programming languages and platforms.\n\n"
    "Education\n"
    "B.S. in Computer Science\n"
    "University of California, Berkeley\n\n"
    "Experience\n"
    "Software Engineer\n"
    "Google\n\n"
    "Skills\n"
    "Python, Java, C++\n\n"
    "Projects\n"
    "Project 1\n"
    "Project 2\n\n"
    "Certifications\n"
    "Certification 1\n"
    "Certification 2"
)

encoded_input = dict(tokenizer(sample_text, return_tensors='tf', padding=True, truncation=True))

_ = model(encoded_input)

model.save("app/models/model_1.0.0.keras", save_format="keras")

# # Define a serving function with an explicit input signature.
# @tf.function(input_signature=[{
#     "input_ids": tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
#     "attention_mask": tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask"),
#     "token_type_ids": tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="token_type_ids")
# }])
# def serve_fn(inputs):
#     token_logits, overall_score = model(inputs)
#     return {"token_logits": token_logits, "overall_score": overall_score}

# # Get the concrete function.
# concrete_fn = serve_fn.get_concrete_function()

# model.save("models/model_1.0.0", save_format="tf", signatures={"serving_default": concrete_fn})