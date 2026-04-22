/// <reference types="vite/client" />

// Allow importing JSON as modules (see src/mock/chest_pain_mid_case.transcript.json).
declare module "*.json" {
  const value: unknown;
  export default value;
}
