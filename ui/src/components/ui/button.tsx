import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

// shadcn-style variant button. Authored locally (rules.md §1.1: fresh code).
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-0",
  {
    variants: {
      variant: {
        default:
          "bg-fg text-bg hover:bg-fg/90 border border-fg/10",
        outline:
          "border border-border-strong bg-transparent text-fg hover:bg-surface-2",
        ghost: "bg-transparent text-fg-muted hover:text-fg hover:bg-surface-2",
        danger:
          "border border-status-dismissed/60 bg-status-dismissed/15 text-status-dismissed hover:bg-status-dismissed/25",
      },
      size: {
        sm: "h-7 px-2.5 text-xs",
        md: "h-9 px-3.5",
        lg: "h-10 px-4",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  ),
);
Button.displayName = "Button";
