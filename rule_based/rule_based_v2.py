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
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stop_words: set[str] = set(stopwords.words('english'))
        
        self.load_data()
        self.build_response_dictionaries()

    def load_data(self) -> None:
        """
        Load the conversations dataset and create a message lookup dictionary.
        If the message lookup dictionary does not exist, it is created and saved.
        
        The conversations dictionary contains the full chat history and the
        message lookup provides access to individual messages by their ID.
        """
        with open('./processed_data/processed_conversations.pkl', 'rb') as f:
            self.conversations_dict: dict[str, list[dict]] = pickle.load(f)

        try:
            with open('./processed_data/message_lookup.pkl', 'rb') as f:
                self.message_lookup: dict[str, dict] = pickle.load(f)

        except FileNotFoundError:
            self.message_lookup = {}
            for _, messages in self.conversations_dict.items():
                for message in messages:
                    self.message_lookup[message['id']] = message
                    
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
        text = text.lower()
        tokens = wordpunct_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token.isalnum()]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def get_keywords(self, text: str) -> list[str]:
        """
        Extract meaningful keywords from the input text by removing common stopwords.
        Stopwords are frequent words like 'the', 'is', 'at' that typically don't
        carry significant meaning.

        Args:
            text: The input text to extract keywords from.

        Returns:
            A list of keywords with stopwords removed.
        """
        tokens = word_tokenize(text)
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
        conversation_pairs: list[tuple[str, str]] = []
        for conv_id, messages in self.conversations_dict.items():
            for message in messages:
                if 'text' in message and message.get('reply_to'):
                    reply_to_id = message['reply_to']
                    if reply_to_id in self.message_lookup:
                        user_message = self.message_lookup[reply_to_id]['text']
                        bot_response = message['text']
                        conversation_pairs.append((user_message, bot_response))

        self.response_dict: dict[str, str] = {}
        self.keyword_dict: dict[str, list[str]] = {}
        
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
        preprocessed_input = self.preprocess(user_input)
        if preprocessed_input in self.response_dict:
            return self.response_dict[preprocessed_input]
        else:
            keywords = self.get_keywords(preprocessed_input)
            possible_responses: list[str] = []
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    possible_responses.extend(self.keyword_dict[keyword])
            if possible_responses:
                return max(set(possible_responses), key=possible_responses.count)
            else:
                return self.generate_fallback_response(user_input)

    def generate_fallback_response(self, user_input: str) -> str:
        """
        Generate a fallback response when no suitable response is found.
        Randomly selects from a list of default responses to provide variety.
        
        Args:
            user_input (str): The original user input that couldn't be matched.
            
        Returns:
            str: A randomly selected default response.
        """
        fallback_responses = [
            "I'm sorry, I didn't quite catch that. Could you please rephrase?",
            "I'm not sure I understand. Could you try explaining that differently?",
            "Hmm, I'm having trouble following. Please rephrase that?",
            "I'm still learning and that's a bit unclear to me. Could you say it another way?",
            "I don't quite understand what you mean. Could you elaborate?",
            "That's not something I'm familiar with. Could you try expressing it differently?"
        ]
        
        return random.choice(fallback_responses)

