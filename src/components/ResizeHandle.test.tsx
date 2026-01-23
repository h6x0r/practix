import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { ResizeHandle } from './ResizeHandle';

describe('ResizeHandle', () => {
  it('should render', () => {
    const onMouseDown = vi.fn();
    render(<ResizeHandle onMouseDown={onMouseDown} isResizing={false} />);

    // The component renders a div - let's find it by its class
    const handle = document.querySelector('.cursor-col-resize');
    expect(handle).toBeInTheDocument();
  });

  it('should call onMouseDown when clicked', () => {
    const onMouseDown = vi.fn();
    render(<ResizeHandle onMouseDown={onMouseDown} isResizing={false} />);

    const handle = document.querySelector('.cursor-col-resize');
    fireEvent.mouseDown(handle!);

    expect(onMouseDown).toHaveBeenCalledTimes(1);
  });

  it('should have different styles when resizing', () => {
    const onMouseDown = vi.fn();
    const { rerender } = render(
      <ResizeHandle onMouseDown={onMouseDown} isResizing={false} />
    );

    let handle = document.querySelector('.cursor-col-resize');
    expect(handle).not.toHaveClass('bg-brand-500');

    rerender(<ResizeHandle onMouseDown={onMouseDown} isResizing={true} />);

    handle = document.querySelector('.cursor-col-resize');
    expect(handle).toHaveClass('bg-brand-500');
  });

  it('should pass mouse event to handler', () => {
    const onMouseDown = vi.fn();
    render(<ResizeHandle onMouseDown={onMouseDown} isResizing={false} />);

    const handle = document.querySelector('.cursor-col-resize');
    fireEvent.mouseDown(handle!, { clientX: 100, clientY: 200 });

    expect(onMouseDown).toHaveBeenCalledWith(
      expect.objectContaining({
        clientX: 100,
        clientY: 200,
      })
    );
  });

  it('should be hidden on mobile (md:flex)', () => {
    const onMouseDown = vi.fn();
    render(<ResizeHandle onMouseDown={onMouseDown} isResizing={false} />);

    const handle = document.querySelector('.cursor-col-resize');
    expect(handle).toHaveClass('hidden');
    expect(handle).toHaveClass('md:flex');
  });
});
