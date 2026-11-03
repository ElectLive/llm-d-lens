// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React from 'react';
import { ChevronDown, CircleHelp } from 'lucide-react';
import { cn } from '../lib/cn';

const fieldBase =
    'w-full rounded-lg border bg-white dark:bg-slate-900 border-[#cbd5e1] dark:border-slate-700 hover:border-[#94a3b8] dark:hover:border-slate-600 ' +
    'px-3 py-2 text-sm text-theme-text placeholder:text-theme-muted transition-colors focus:bg-white dark:focus:bg-slate-900 ' +
    'focus:outline-none focus:ring-2 focus:ring-[#037ffc]/30 focus:border-[#037ffc] dark:focus:ring-emerald-500/40 dark:focus:border-emerald-500 ' +
    'disabled:opacity-50 disabled:cursor-not-allowed';

const fieldVariants = {
    toolbar: 'h-8 rounded-lg border-[#cbd5e1] bg-transparent py-1.5 text-xs dark:border-transparent dark:bg-transparent hover:bg-theme-bg/70 focus:bg-theme-bg focus:ring-1 focus:ring-sky-500/30 focus:border-sky-500/40',
};

export function Label({ className, children, help, ...props }) {
    const label = (
        <label
            className={cn('block text-xs font-semibold text-slate-600 dark:text-slate-400', help ? 'mb-0' : 'mb-1.5', className)}
            {...props}
        >
            {children}
        </label>
    );
    if (!help) return label;
    return (
        <div className="relative mb-1.5 flex items-center gap-1.5">
            {label}
            <HelpTooltip label={`${typeof children === 'string' ? children : 'Field'} help`}>
                {help}
            </HelpTooltip>
        </div>
    );
}

export const Input = React.forwardRef(function Input({ className, error, icon: Icon, variant, ...props }, ref) {
    if (!Icon) {
        return (
            <input
                ref={ref}
                className={cn(fieldBase, fieldVariants[variant], error && 'border-red-500 focus:border-red-500 focus:ring-red-500/40', className)}
                {...props}
            />
        );
    }
    return (
        <div className="relative">
            <Icon
                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-theme-muted"
                aria-hidden="true"
            />
            <input
                ref={ref}
                className={cn(
                    fieldBase,
                    fieldVariants[variant],
                    'pl-9',
                    error && 'border-red-500 focus:border-red-500 focus:ring-red-500/40',
                    className
                )}
                {...props}
            />
        </div>
    );
});

export const Select = React.forwardRef(function Select({ className, error, children, icon: Icon, variant, ...props }, ref) {
    const decorated = Icon || variant === 'toolbar';
    const select = (
        <select
            ref={ref}
            className={cn(fieldBase, fieldVariants[variant], 'pr-8', decorated && 'appearance-none', Icon && 'pl-9',
                error && 'border-red-500 focus:border-red-500 focus:ring-red-500/40', className)}
            {...props}
        >
            {children}
        </select>
    );
    if (!decorated) return select;
    return (
        <div className="relative">
            {Icon && <Icon className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-theme-muted" aria-hidden="true" />}
            {select}
            <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3 w-3 -translate-y-1/2 text-theme-muted" aria-hidden="true" />
        </div>
    );
});

export const Textarea = React.forwardRef(function Textarea({ className, error, rows = 4, ...props }, ref) {
    return (
        <textarea
            ref={ref}
            rows={rows}
            className={cn(fieldBase, 'font-mono text-xs leading-relaxed', error && 'border-red-500 focus:border-red-500 focus:ring-red-500/40', className)}
            {...props}
        />
    );
});

export const Checkbox = React.forwardRef(function Checkbox({ className, label, ...props }, ref) {
    const box = (
        <input
            ref={ref}
            type="checkbox"
            className={cn(
                'h-4 w-4 shrink-0 rounded border-slate-300 dark:border-slate-600 accent-emerald-600',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/40',
                className
            )}
            {...props}
        />
    );
    if (!label) return box;
    return (
        <label className="inline-flex items-center gap-2 text-sm text-theme-text cursor-pointer select-none">
            {box}
            <span>{label}</span>
        </label>
    );
});

export function HelpTooltip({ label, children, trigger, className, iconSize = 14, tooltipClassName = 'inset-x-0 top-full pt-1' }) {
    const helpId = React.useId();
    return (
        <span className={cn('group inline-flex', className)}>
            {trigger ? React.cloneElement(trigger, { 'aria-describedby': helpId }) : <button type="button" aria-label={label}
                aria-describedby={helpId}
                className="inline-flex rounded-full p-0.5 text-theme-muted hover:text-theme-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/40">
                <CircleHelp size={iconSize} aria-hidden="true" />
            </button>}
            <span className={cn('invisible absolute z-10 group-hover:visible group-focus-within:visible', tooltipClassName)}>
                <span id={helpId} role="tooltip"
                    className="block rounded-lg border border-theme-border bg-theme-card p-3 text-xs font-normal leading-relaxed text-theme-text shadow-lg">
                    {children}
                </span>
            </span>
        </span>
    );
}
