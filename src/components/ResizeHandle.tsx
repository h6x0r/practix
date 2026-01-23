import React from 'react';

interface ResizeHandleProps {
  onMouseDown: (e: React.MouseEvent) => void;
  isResizing: boolean;
}

/**
 * Draggable resize handle for resizable panels.
 * Shows visual feedback on hover and during drag.
 */
export const ResizeHandle: React.FC<ResizeHandleProps> = ({ onMouseDown, isResizing }) => {
  return (
    <div
      className={`
        hidden md:flex
        w-1 h-full cursor-col-resize
        group relative z-10
        transition-colors duration-150
        ${isResizing ? 'bg-brand-500' : 'bg-gray-200 dark:bg-dark-border hover:bg-brand-400'}
      `}
      onMouseDown={onMouseDown}
    >
      {/* Visual indicator dots (shown on hover) */}
      <div
        className={`
          absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
          flex flex-col gap-1
          opacity-0 group-hover:opacity-100 transition-opacity
          ${isResizing ? 'opacity-100' : ''}
        `}
      >
        <div className="w-1 h-1 rounded-full bg-gray-400 dark:bg-gray-500" />
        <div className="w-1 h-1 rounded-full bg-gray-400 dark:bg-gray-500" />
        <div className="w-1 h-1 rounded-full bg-gray-400 dark:bg-gray-500" />
      </div>

      {/* Wider hit area for easier grabbing */}
      <div className="absolute inset-y-0 -left-1 -right-1" />
    </div>
  );
};
