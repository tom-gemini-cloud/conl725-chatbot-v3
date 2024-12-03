import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import json
import pickle
import os
import ssl
from tqdm import tqdm
import random
# Download required NLTK data ignoring SSL errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class NLTKDialogPreprocessor:
    """A class for preprocessing dialogue text using NLTK tools.
    
    This class provides functionality for loading, preprocessing, and analysing conversational
    text data using NLTK. It handles text cleaning, tokenisation, lemmatisation, and
    vocabulary building.
    
    Attributes:
        lemmatizer: WordNetLemmatizer instance for word lemmatisation
        stop_words: Set of English stop words from NLTK
        conversations: Dictionary storing conversations indexed by conversation_id
        vocabulary: FreqDist object containing word frequency distribution
    """

    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')      # For sentence and word tokenisation
        nltk.download('stopwords')  # For common words filtering (e.g., 'the', 'is', 'at')
        nltk.download('wordnet')    # For word lemmatisation (reducing words to base form)
        
        self.lemmatiser = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.conversations = {}  # Dictionary to store conversations by conversation_id
        self.vocabulary = FreqDist()

    def load_json_conversations(self, json_data):
        """Load and organise conversations from JSON-formatted data.
        
        Args:
            json_data (str): JSON-formatted string containing conversation data,
                           with one JSON object per line
        
        The method expects each JSON object to have the following structure:
            {
                'conversation_id': str,
                'id': str,
                'text': str,
                'speaker': str,
                'reply-to': str,
                'meta': {'parsed': list}
            }
        """
        print("Loading conversations...")
        
        # Split data into lines
        lines = [line for line in json_data.split('\n') if line.strip()]
        
        # Group messages by conversation_id with progress bar
        for line in tqdm(lines, desc="Processing conversations"):
            message = json.loads(line)
            conv_id = message['conversation_id']
            
            if conv_id not in self.conversations:
                self.conversations[conv_id] = []
                
            self.conversations[conv_id].append({
                'id': message['id'],
                'text': message['text'],
                'speaker': message['speaker'],
                'reply_to': message['reply-to'],
                'parsed': message['meta']['parsed']
            })
        
        print(f"Loaded {len(self.conversations)} conversations")

    def clean_text(self, text):
        """Perform basic text cleaning operations.
        
        Args:
            text (str): Raw input text to be cleaned
            
        Returns:
            str: Cleaned text with lowercase conversion and basic tokenisation
            
        Example:
            >>> preprocessor.clean_text("Hello, World!")
            "hello , world !"
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = ' '.join(word_tokenize(text))
        
        return text.strip()

    def preprocess_text(self, text):
        """Execute the complete text preprocessing pipeline.
        
        Applies the following steps:
        1. Text cleaning
        2. Tokenisation
        3. Stop word removal
        4. Lemmatisation
        
        Args:
            text (str): Raw input text to be processed
            
        Returns:
            str: Processed text with tokens joined by spaces
            
        Example:
            >>> preprocessor.preprocess_text("The cats are running quickly")
            "cat run quickly"
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenise
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatise
        processed_tokens = [
            self.lemmatiser.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(processed_tokens)

    def build_vocabulary(self):
        """Create a frequency distribution of words from processed texts.
        
        Analyses all processed texts in the conversations and constructs a
        vocabulary with word frequencies. The progress is displayed via a
        progress bar for each conversation being processed.
        
        Updates:
            self.vocabulary: FreqDist object containing word frequency distribution
        
        Note:
            This method should be called after processing texts using
            process_dataset(). The vocabulary size is printed upon completion.
        """
        all_words = []
        for conv_id, messages in tqdm(self.conversations.items(), desc="Building vocabulary"):
            for message in messages:
                if 'processed_text' in message:
                    words = word_tokenize(message['processed_text'])
                    all_words.extend(words)
        
        self.vocabulary = FreqDist(all_words)
        print(f"Vocabulary size: {len(self.vocabulary)}")

    def process_dataset(self):
        """Execute the complete preprocessing pipeline on all conversations.
        
        Performs the following operations with progress visualisation:
        1. Flattens conversations into a list of messages
        2. Processes all message texts using preprocess_text()
        3. Extracts POS tags from parsed data
        4. Builds vocabulary using build_vocabulary()
        
        The method displays progress bars for:
        - Message processing
        - Vocabulary building
        
        Note:
            This is the main processing method that should be called after
            loading conversations.
        """
        print("Starting preprocessing pipeline...")
        
        # Process all messages with progress bar
        messages_list = [
            (conv_id, message)
            for conv_id, messages in self.conversations.items()
            for message in messages
        ]
        
        for conv_id, message in tqdm(messages_list, desc="Processing messages"):
            # Process text
            message['processed_text'] = self.preprocess_text(message['text'])
            
            # Use POS tags from parsed data
            if message['parsed']:
                message['pos_tags'] = [
                    (tok['tok'], tok['tag'])
                    for sent in message['parsed']
                    for tok in sent['toks']
                ]
        
        # Build vocabulary
        self.build_vocabulary()
    
    def save_data(self, output_path):
        """Save processed conversations and vocabulary to disk.
        
        Args:
            output_path (str): Directory path where files should be saved
            
        Saves the following files:
        - processed_conversations.pkl: Pickle file of processed conversations
        - processed_conversations.json: JSON file of processed conversations
        - vocabulary.pkl: Pickle file of vocabulary FreqDist
        - vocabulary.json: JSON file of vocabulary frequencies
        
        Note:
            Creates the output directory if it doesn't exist.
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save conversations in both pickle and JSON formats
        with open(os.path.join(output_path, 'processed_conversations.pkl'), 'wb') as f:
            pickle.dump(self.conversations, f)
        
        with open(os.path.join(output_path, 'processed_conversations.json'), 'w') as f:
            json.dump(self.conversations, f, indent=2)
        
        # Save vocabulary in both pickle and JSON formats
        with open(os.path.join(output_path, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(self.vocabulary, f)
        
        # Convert FreqDist to dictionary for JSON serialization
        vocab_dict = dict(self.vocabulary)
        with open(os.path.join(output_path, 'vocabulary.json'), 'w') as f:
            json.dump(vocab_dict, f, indent=2)
        
        print(f"Saved processed data to {output_path}")

# Main execution block
if __name__ == "__main__":
    # Initialise preprocessor
    preprocessor = NLTKDialogPreprocessor()
    
    # Load and process the dataset
    with open('./data/movie_corpus/utterances.jsonl', 'r') as f:
        json_data = f.read()
    
    preprocessor.load_json_conversations(json_data)
    preprocessor.process_dataset()
    
    # Save the processed data
    preprocessor.save_data('processed_data')
    
    # Print a randomly selected sample conversation
    sample_conv_id = list(preprocessor.conversations.keys())[random.randint(0, len(preprocessor.conversations) - 1)]
    print("\nSample processed conversation:")
    for message in preprocessor.conversations[sample_conv_id]:
        print(f"Conversation: {sample_conv_id}")
        print(f"Speaker {message['speaker']}:")
        print(f"Original: {message['text']}")
        print(f"Processed: {message['processed_text']}")
        print(f"POS Tags: {message['pos_tags']}\n")