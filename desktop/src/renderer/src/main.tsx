import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ensureBridge } from './bridge';
import './styles.css';

/* Electron: preload bridge. Browser preview: honest read-only stub. */
ensureBridge();

const root = document.getElementById('root');
if (!root) throw new Error('Renderer root not found');

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
