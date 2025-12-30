// Translation configuration
export const TRANSLATION_CONFIG = {
  API_KEY: process.env.GOOGLE_TRANSLATE_API_KEY || '',
  TARGET_LANGUAGE: 'ur', // Urdu
  SOURCE_LANGUAGE: 'en', // English
  API_ENDPOINT: 'https://translation.googleapis.com/language/translate/v2',
};

// Function to check if API key is configured
export const isApiKeyConfigured = (): boolean => {
  return TRANSLATION_CONFIG.API_KEY !== '' && TRANSLATION_CONFIG.API_KEY !== 'YOUR_API_KEY_HERE';
};

// Function to get API key
export const getApiKey = (): string => {
  return TRANSLATION_CONFIG.API_KEY;
}; 
