declare module "monaco-editor" {
  export namespace editor {
    interface IStandaloneCodeEditor {
      getValue(): string;
      setValue(value: string): void;
      getModel(): ITextModel | null;
      onDidChangeModelContent(listener: (e: unknown) => void): IDisposable;
      layout(dimension?: { width: number; height: number }): void;
      focus(): void;
      addCommand(keybinding: number, handler: () => void): string | null;
    }

    interface ITextModel {
      getValue(): string;
      getLineCount(): number;
    }

    interface IEditorOptions {
      fontSize?: number;
      fontFamily?: string;
      minimap?: { enabled?: boolean };
      lineNumbers?: "on" | "off" | "relative";
      theme?: string;
      [key: string]: unknown;
    }
  }

  export interface IDisposable {
    dispose(): void;
  }
}
