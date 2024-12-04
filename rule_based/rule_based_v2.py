import pickle
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random

class Chatbot:
    def __init__(self) -> None:
        """
        Initialise the RuleBasedChatbot by downloading necessary NLTK data files,
        initialising preprocessing tools, and loading data to build response dictionaries.
        """
        # Download necessary NLTK data files
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        # Initialise lemmatiser and stop words set
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stop_words: set[str] = set(stopwords.words('english'))
        
        # Load data and build response dictionaries
        self.load_data()
        self.build_response_dictionaries()

    def load_data(self) -> None:
        """
        Load the conversations dataset and create a message lookup dictionary.
        If the message lookup dictionary does not exist, it is created and saved.
        
        The conversations dictionary contains the full chat history and the
        message lookup provides access to individual messages by their ID.
        """
        # Load conversations dictionary from a Pickle file
        with open('./processed_data/processed_conversations.pkl', 'rb') as f:
            self.conversations_dict: dict[str, list[dict]] = pickle.load(f)

        try:
            # Attempt to load the message lookup dictionary
            with open('./processed_data/message_lookup.pkl', 'rb') as f:
                self.message_lookup: dict[str, dict] = pickle.load(f)

        except FileNotFoundError:
            # If not found, create the message lookup dictionary
            self.message_lookup = {}
            for _, messages in self.conversations_dict.items():
                for message in messages:
                    self.message_lookup[message['id']] = message
                    
            # Save the newly created message lookup dictionary
            with open('./processed_data/message_lookup.pkl', 'wb') as f:
                pickle.dump(self.message_lookup, f)

    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by converting it to lowercase,
        tokenising (including punctuation removal), and lemmatising the words.

        Args:
            text: The input text to preprocess.

        Returns:
            The preprocessed text as a single string with standardised tokens.
        """
        # Convert text to lowercase
        text = text.lower()
        # Tokenise text, including punctuation removal
        tokens = wordpunct_tokenize(text)
        # Lemmatise tokens and remove non-alphanumeric tokens
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token.isalnum()]
        # Join tokens into a single string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def get_keywords(self, text: str) -> list[str]:
        """
        Extract keywords from the input text by removing common stopwords.

        Args:
            text: The input text to extract keywords from.

        Returns:
            A list of keywords with stopwords removed.
        """
        # Tokenise the input text
        tokens = word_tokenize(text)
        # Remove stopwords from the tokens
        keywords = [word for word in tokens if word not in self.stop_words]
        return keywords

    def build_response_dictionaries(self) -> None:
        """
        Build response and keyword dictionaries from conversation pairs.
        
        Creates two main dictionaries:
        1. response_dict: Maps preprocessed inputs directly to responses
        2. keyword_dict: Maps individual keywords to possible responses,
                         used as a fallback when exact matches aren't found
        """
        # Extract conversation pairs from the conversations dictionary
        conversation_pairs: list[tuple[str, str]] = []
        for conv_id, messages in self.conversations_dict.items():
            for message in messages:
                if 'text' in message and message.get('reply_to'):
                    reply_to_id = message['reply_to']
                    if reply_to_id in self.message_lookup:
                        user_message = self.message_lookup[reply_to_id]['text']
                        bot_response = message['text']
                        conversation_pairs.append((user_message, bot_response))

        # Initialise response and keyword dictionaries
        self.response_dict: dict[str, str] = {}
        self.keyword_dict: dict[str, list[str]] = {}
        
        # Populate the dictionaries with conversation pairs
        for user_input, bot_response in conversation_pairs:
            preprocessed_input = self.preprocess(user_input)
            self.response_dict[preprocessed_input] = bot_response
            
            keywords = self.get_keywords(preprocessed_input)
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    self.keyword_dict[keyword].append(bot_response)
                else:
                    self.keyword_dict[keyword] = [bot_response]

    def get_response(self, user_input: str) -> str:
        """
        Get a response for the given user input using a two-step matching process:
        1. Try to find an exact match with a preprocessed previous input
        2. If no exact match, look for responses associated with matching keywords
           and return the most common response

        Args:
            user_input: The user's input message.

        Returns:
            The chatbot's response, either matched exactly or based on keywords.
        """
        # Preprocess the user input
        preprocessed_input = self.preprocess(user_input)
        # Check for an exact match in the response dictionary
        if preprocessed_input in self.response_dict:
            return self.response_dict[preprocessed_input]
        else:
            # If no exact match, find responses based on keywords
            keywords = self.get_keywords(preprocessed_input)
            possible_responses: list[str] = []
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    possible_responses.extend(self.keyword_dict[keyword])
            # Return the most common response if possible responses exist
            if possible_responses:
                return max(set(possible_responses), key=possible_responses.count)
            else:
                # Generate a fallback response if no matches are found
                return self.generate_fallback_response(user_input)

    def generate_fallback_response(self) -> str:
        """
        Generate a fallback response when no suitable response is found.
        Randomly selects from a list of default responses to provide variety.
        
        Returns:
            str: A randomly selected default response.
        """
        # List of fallback responses to use when no match is found
        fallback_responses = [
            "I'm sorry, I didn't quite catch that. Could you please rephrase?",
            "I'm not sure I understand. Could you try explaining that differently?",
            "Hmm, I'm having trouble following. Do you mind rephrasing that?",
            "I'm still learning and that's a bit unclear to me. Could you say it another way?",
            "I don't quite understand what you mean. Could you elaborate?",
            "That's not something I'm familiar with. Could you try expressing it differently?"
        ]
        
        # Randomly select and return one of the fallback responses
        return random.choice(fallback_responses)

