export function Header() {
  return (
    <header className="h-14 border-b border-border flex items-center justify-between px-4 bg-card">
      <h2 className="text-sm font-semibold tracking-tight">Persi</h2>
      <div className="flex items-center gap-3">
        <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-1 rounded border border-border bg-muted px-1.5 text-[10px] font-medium text-muted-foreground">
          <span className="text-xs">⌘</span>K
        </kbd>
        <span className="text-xs text-muted-foreground">v1.0</span>
      </div>
    </header>
  );
}
