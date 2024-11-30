import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from typing import Dict, List, Set, Tuple, Optional

class Chatbot:
    def __init__(self, data_path: str):
        # Initialize NLTK components
        self._download_nltk_dependencies()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess,
            stop_words=None,
            ngram_range=(1, 3)
        )
        
        # Load and process data
        self.conversations_dict = self._load_data(data_path)
        self.message_lookup = self._build_message_lookup()
        self.conversation_pairs = self._extract_conversation_pairs()
        self.response_vectors = None
        self.keyword_dict = {}
        self.context_history: List[Tuple[str, str]] = []
        
        # Build response database
        self._build_response_database()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _download_nltk_dependencies(self):
        """Download necessary NLTK data files for text processing.

        This method attempts to download the required NLTK data files
        for tokenisation, lemmatisation, and stopword removal. If a
        download fails, an error is logged.
        """
        dependencies = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        for dependency in dependencies:
            try:
                nltk.download(dependency, quiet=True)
            except Exception as e:
                logging.error(f"Failed to download {dependency}: {str(e)}")

    def _load_data(self, data_path: str) -> Dict:
        """Load conversation data from a pickle file.

        Args:
            data_path (str): The file path to the pickle file containing conversation data.

        Returns:
            Dict: A dictionary containing the loaded conversation data.
                  Returns an empty dictionary if loading fails.
        """
        try:
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            return {}

    def _build_message_lookup(self) -> Dict:
        """Create a lookup dictionary for messages by their IDs.

        Returns:
            Dict: A dictionary mapping message IDs to message content.
        """
        message_lookup = {}
        for conv_id, messages in self.conversations_dict.items():
            for message in messages:
                message_lookup[message['id']] = message
        return message_lookup

    def _extract_conversation_pairs(self) -> List[Tuple[str, str]]:
        """Extract pairs of user messages and bot responses from the dataset.

        Returns:
            List[Tuple[str, str]]: A list of tuples, each containing a user message
                                   and the corresponding bot response.
        """
        pairs = []
        for conv_id, messages in self.conversations_dict.items():
            for message in messages:
                if 'text' in message and message.get('reply_to'):
                    reply_to_id = message['reply_to']
                    if reply_to_id in self.message_lookup:
                        user_message = self.message_lookup[reply_to_id]['text']
                        bot_response = message['text']
                        pairs.append((user_message, bot_response))
        return pairs

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text by cleaning and normalising it.

        This includes converting text to lowercase, removing punctuation,
        tokenising, lemmatising, and removing stopwords and short tokens.

        Args:
            text (str): The text to preprocess.

        Returns:
            List[str]: A list of processed tokens.
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return tokens

    def _build_response_database(self):
        """Build a response database using TF-IDF vectorisation.

        This method prepares conversation texts, fits the TF-IDF vectoriser,
        and constructs a keyword dictionary for context-based responses.
        """
        # Prepare conversation texts
        conversation_texts = [pair[0] for pair in self.conversation_pairs]
        
        # Fit and transform the vectorizer
        self.response_vectors = self.vectorizer.fit_transform(conversation_texts)
        
        # Build keyword dictionary with context
        for user_input, bot_response in self.conversation_pairs:
            keywords = self.preprocess(user_input)
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    self.keyword_dict[keyword].append((bot_response, user_input))
                else:
                    self.keyword_dict[keyword] = [(bot_response, user_input)]

    def _get_response_similarity(self, user_input: str) -> Tuple[str, float]:
        """Calculate the similarity of user input to stored responses.

        Uses TF-IDF and cosine similarity to find the most similar response.

        Args:
            user_input (str): The user's input text.

        Returns:
            Tuple[str, float]: The most similar bot response and its similarity score.
                               Returns an empty string and 0.0 if no suitable response is found.
        """
        input_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vector, self.response_vectors).flatten()
        
        # Get the most similar response
        max_sim_idx = similarities.argmax()
        max_similarity = similarities[max_sim_idx]
        
        if max_similarity > 0.3:  # Threshold for acceptable similarity
            return self.conversation_pairs[max_sim_idx][1], max_similarity
        return "", 0.0

    def _get_context_based_response(self, user_input: str) -> Optional[str]:
        """Generate a response based on recent conversation context.

        Considers the last few exchanges to enhance the input with context.

        Args:
            user_input (str): The user's input text.

        Returns:
            Optional[str]: A contextually relevant response, or None if no suitable response is found.
        """
        if not self.context_history:
            return None
            
        # Consider last 3 exchanges for context
        recent_context = ' '.join([msg for exchange in self.context_history[-3:] for msg in exchange])
        context_enhanced_input = f"{recent_context} {user_input}"
        
        response, similarity = self._get_response_similarity(context_enhanced_input)
        if similarity > 0.4:  # Higher threshold for context-based responses
            return response
        return None

    def get_response(self, user_input: str) -> str:
        """Generate a response to user input using multiple strategies.

        Attempts context-based, similarity-based, and keyword-based responses
        in that order, falling back to a default message if necessary.

        Args:
            user_input (str): The user's input text.

        Returns:
            str: The generated bot response.
        """
        # Add basic greeting handling
        greetings = {'hello', 'hi', 'hey', 'greetings'}
        if user_input.lower().strip() in greetings:
            return "Hello! How can I help you today?"
        
        try:
            # Try context-based response first
            context_response = self._get_context_based_response(user_input)
            if context_response:
                self.context_history.append((user_input, context_response))
                return context_response

            # Try similarity-based response
            response, similarity = self._get_response_similarity(user_input)
            if response:
                self.context_history.append((user_input, response))
                return response

            # Fall back to keyword-based response
            keywords = self.preprocess(user_input)
            possible_responses = []
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    possible_responses.extend(self.keyword_dict[keyword])

            if possible_responses:
                # Select response based on both frequency and context similarity
                best_response = max(possible_responses, 
                                  key=lambda x: (possible_responses.count(x[0]), 
                                               self._calculate_context_relevance(x[1], user_input)))
                self.context_history.append((user_input, best_response[0]))
                return best_response[0]

            # Fallback response
            return "I apologise, but I don't quite understand. Could you please rephrase that or provide more context?"

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I encountered an error. Please try again."

    def _calculate_context_relevance(self, stored_context: str, current_input: str) -> float:
        """Calculate the relevance of stored context to the current input.

        Uses cosine similarity to measure how relevant the stored context is
        to the current user input.

        Args:
            stored_context (str): The stored context from previous exchanges.
            current_input (str): The current user input.

        Returns:
            float: The relevance score between 0 and 1.
        """
        try:
            vec1 = self.vectorizer.transform([stored_context])
            vec2 = self.vectorizer.transform([current_input])
            return cosine_similarity(vec1, vec2)[0][0]
        except Exception:
            return 0.0

    def run(self):
        """Run the chatbot interface for user interaction.

        This method starts an interactive session where the user can
        input text and receive responses from the chatbot. The session
        continues until the user types 'exit', 'quit', or 'bye'.
        """
        print("Enhanced Chatbot: Hello! How can I assist you today? (Type 'exit' to quit)")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Bot: Goodbye! Have a great day!")
                    break
                    
                response = self.get_response(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye! Have a great day!")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {str(e)}")
                print("Bot: I encountered an error. Please try again.")

# Usage
if __name__ == "__main__":
    chatbot = Chatbot('./processed_data/processed_conversations.pkl')
    chatbot.run()