import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotWidget from './ChatbotWidget';
import TranslationButton from '../components/TranslationButton';


export default function LayoutWrapper(props) {
  return (

      <OriginalLayout {...props}>
        {props.children}
        <ChatbotWidget />
        <TranslationButton />
      </OriginalLayout>

  );
}