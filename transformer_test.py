import streamlit as st
import transformers
from transformers import pipeline
from transformers import set_seed

# Set up the app title
st.title("Transformers Playground")

# Sidebar for task selection
st.sidebar.title("Select Task")
task = st.sidebar.selectbox(
    "Choose a task:",
    [
        "Sentiment Analysis",
        "Zero-Shot Classification",
        "Text Generation",
        "Text Generation (Seed Set)",
        "Fill-Mask",
        "Named Entity Recognition",
        "Question Answering",
        "Text Summarization",
        "Translation",
    ],
)

# Load models based on the selected task
@st.cache_resource  # Cache models to avoid reloading
def load_model(task):
    if task == "Sentiment Analysis":
        return pipeline('sentiment-analysis')
    elif task == "Zero-Shot Classification":
        return pipeline('zero-shot-classification')
    elif task == "Text Generation":
        return pipeline('text-generation', model='gpt2')
    elif task == "Text Generation (Seed Set)":
        set_seed(42)
        return pipeline('text-generation', model='gpt2')
    elif task == "Fill-Mask":
        return pipeline('fill-mask')
    elif task == "Named Entity Recognition":
        return pipeline('ner', grouped_entities=True)
    elif task == "Question Answering":
        return pipeline('question-answering')
    elif task == "Text Summarization":
        return pipeline('summarization')
    elif task == "Translation":
        return pipeline('translation_en_to_fr', model='t5-small')
    else:
        return None

model = load_model(task)

# Main app logic
if task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    user_input = st.text_area("Enter text for sentiment analysis:", "I love using Streamlit!")
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = model(user_input)[0]
            st.write(f"**Sentiment:** {result['label']}")
            st.write(f"**Confidence Score:** {result['score']:.4f}")

elif task == "Zero-Shot Classification":
    st.header("Zero-Shot Classification")
    user_input = st.text_area("Enter text to classify:", "I love playing football and watching movies.")
    labels = st.text_input("Enter possible labels (comma-separated):", "sports, entertainment, politics")
    if st.button("Classify"):
        if user_input.strip() == "" or labels.strip() == "":
            st.warning("Please enter text and labels.")
        else:
            labels = [label.strip() for label in labels.split(",")]
            result = model(user_input, candidate_labels=labels)
            st.write(f"**Text:** {user_input}")
            st.write(f"**Labels:** {labels}")
            st.write(f"**Predicted Label:** {result['labels'][0]}")
            st.write(f"**Scores:** {result['scores']}")

elif task == "Text Generation":
    st.header("Text Generation")
    user_input = st.text_area("Enter a prompt for text generation:", "Once upon a time")
    if st.button("Generate Text"):
        if user_input.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            result = model(user_input, max_length=50, num_return_sequences=1)
            st.write(f"**Generated Text:** {result[0]['generated_text']}")

elif task == "Text Generation (Seed Set)":
    st.header("Text Generation (Seed Set)")
    user_input = st.text_area("Enter a prompt for text generation:", "Once upon a time")
    if st.button("Generate Text"):
        if user_input.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            result = model(user_input, max_length=50, num_return_sequences=1)
            st.write(f"**Generated Text:** {result[0]['generated_text']}")

elif task == "Fill-Mask":
    st.header("Fill-Mask")
    user_input = st.text_area("Enter text with a [MASK] token:", "The capital of France is [MASK].")
    if st.button("Fill Mask"):
        if user_input.strip() == "":
            st.warning("Please enter text with a [MASK] token.")
        else:
            result = model(user_input)
            st.write(f"**Filled Text:** {result[0]['sequence']}")

elif task == "Named Entity Recognition":
    st.header("Named Entity Recognition")
    user_input = st.text_area("Enter text for NER:", "My name is John and I live in New York.")
    if st.button("Extract Entities"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = model(user_input)
            st.write(f"**Entities:** {result}")

elif task == "Question Answering":
    st.header("Question Answering")
    context = st.text_area("Enter context:", "The Eiffel Tower is located in Paris, France.")
    question = st.text_input("Enter question:", "Where is the Eiffel Tower located?")
    if st.button("Answer"):
        if context.strip() == "" or question.strip() == "":
            st.warning("Please enter context and question.")
        else:
            result = model(question=question, context=context)
            st.write(f"**Answer:** {result['answer']}")

elif task == "Text Summarization":
    st.header("Text Summarization")
    user_input = st.text_area("Enter text to summarize:", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.")
    if st.button("Summarize"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = model(user_input, max_length=50, min_length=25, do_sample=False)
            st.write(f"**Summary:** {result[0]['summary_text']}")

elif task == "Translation":
    st.header("Translation (English to French)")
    user_input = st.text_area("Enter text to translate:", "Hello, how are you?")
    if st.button("Translate"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = model(user_input)
            st.write(f"**Translated Text:** {result[0]['translation_text']}")
