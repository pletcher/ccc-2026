from pathlib import Path

from transformers import pipeline

## Results for first successful training on sentences
## Results on held-out test data:
# {'test_loss': 0.3936876654624939,
# 'test_accuracy': 0.9016501650165016,
# 'test_f1': 0.902188927588527,
# 'test_runtime': 86.3297,
# 'test_samples_per_second': 17.549,
# 'test_steps_per_second': 1.1,
# 'epoch': 5.0}
## Results on eval dataset:
# {'eval_loss': '0.3554',
# 'eval_accuracy': '0.9076',
# 'eval_f1': '0.908',
# 'eval_runtime': '86.76',
# 'eval_samples_per_second': '17.46',
# 'eval_steps_per_second': '1.095',
# 'epoch': '5'}

ROOT_DIR = Path(__file__).parent.parent

MODEL = (
    ROOT_DIR
    / "speech-narrative_classification_model_output"
    / "grc-homeric-speech-narrative-sentence-classification"
)

classifier = pipeline(
    "text-classification", model=str(MODEL), aggregation_strategy="simple"
)


tutor_speech = """στάντες δʼ ἵνʼ αὐτοὺς οἱ τεταγμένοι βραβῆς
κλήροις ἔπηλαν καὶ κατέστησαν δίφρους ,
χαλκῆς ὑπαὶ σάλπιγγος ᾖξαν · οἱ δʼ ἅμα
ἵπποις ὁμοκλήσαντες ἡνίας χεροῖν
ἔσεισαν · ἐν δὲ πᾶς ἐμεστώθη δρόμος
κτύπου κροτητῶν ἁρμάτων · κόνις δʼ ἄνω
φορεῖθʼ · ὁμοῦ δὲ πάντες ἀναμεμιγμένοι
φείδοντο κέντρων οὐδέν , ὡς ὑπερβάλοι
χνόας τις αὐτῶν καὶ φρυάγμαθʼ ἱππικά ."""

clytemnestra_speech = """ἔπαιρε δὴ σὺ θύμαθʼ ἡ παροῦσά μοι
πάγκαρπʼ , ἄνακτι τῷδʼ ὅπως λυτηρίους
εὐχὰς ἀνάσχω δειμάτων , ἃ νῦν ἔχω ."""

print(classifier(clytemnestra_speech))


# line classifier:

# {'test_loss': 0.37317410111427307, 'test_accuracy': 0.8603286384976526, 'test_f1': 0.860335625966156}
# Epoch | Training loss | Validation Loss | Accuracy | F1
# 5 | 0.271855   |  0.461055 |    0.852446  |  0.852446
