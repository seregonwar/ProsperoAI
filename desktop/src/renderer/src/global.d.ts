/// <reference types="vite/client" />

import type { ProsperoApi } from '../../shared/types';

declare global {
  interface Window {
    prospero: ProsperoApi;
  }
}

export {};
