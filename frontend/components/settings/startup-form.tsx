"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { Startup } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2, Save, Plus } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const STAGES = [
  "idea",
  "validation",
  "mvp",
  "launch",
  "growth",
  "scale",
  "mature",
];

export function StartupForm() {
  const { toast } = useToast();
  const [startup, setStartup] = useState<Startup | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({
    name: "",
    one_liner: "",
    problem_statement: "",
    target_audience: "",
    stage: "idea",
    industry: "",
    business_model: "",
    context_notes: "",
  });

  useEffect(() => {
    apiFetch<Startup | null>("/profile/startup")
      .then((s) => {
        if (s) {
          setStartup(s);
          setForm({
            name: s.name || "",
            one_liner: s.one_liner || "",
            problem_statement: s.problem_statement || "",
            target_audience: s.target_audience || "",
            stage: s.stage || "idea",
            industry: s.industry || "",
            business_model: s.business_model || "",
            context_notes: s.context_notes || "",
          });
        }
      })
      .finally(() => setLoading(false));
  }, []);

  function updateField(field: string, value: string) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function handleSave() {
    setSaving(true);
    try {
      if (startup) {
        const updated = await apiFetch<Startup>(
          `/profile/startup/${startup.id}`,
          {
            method: "PUT",
            body: JSON.stringify(form),
          }
        );
        setStartup(updated);
        toast({ title: "Startup saved", description: "Your changes have been saved." });
      } else {
        const created = await apiFetch<Startup>("/profile/startup", {
          method: "POST",
          body: JSON.stringify(form),
        });
        setStartup(created);
        toast({ title: "Startup created", description: "Your startup profile has been created." });
      }
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Something went wrong", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="name">Startup Name</Label>
          <Input
            id="name"
            value={form.name}
            onChange={(e) => updateField("name", e.target.value)}
            placeholder="My Startup"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="stage">Stage</Label>
          <Select
            value={form.stage}
            onValueChange={(v) => updateField("stage", v)}
          >
            <SelectTrigger id="stage">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STAGES.map((s) => (
                <SelectItem key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="one_liner">One Liner</Label>
        <Input
          id="one_liner"
          value={form.one_liner}
          onChange={(e) => updateField("one_liner", e.target.value)}
          placeholder="A short description of what you're building"
        />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="industry">Industry</Label>
          <Input
            id="industry"
            value={form.industry}
            onChange={(e) => updateField("industry", e.target.value)}
            placeholder="e.g. SaaS, FinTech, HealthTech"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="business_model">Business Model</Label>
          <Input
            id="business_model"
            value={form.business_model}
            onChange={(e) => updateField("business_model", e.target.value)}
            placeholder="e.g. B2B SaaS, Marketplace"
          />
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="target_audience">Target Audience</Label>
        <Input
          id="target_audience"
          value={form.target_audience}
          onChange={(e) => updateField("target_audience", e.target.value)}
          placeholder="Who are you building for?"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="problem_statement">Problem Statement</Label>
        <Textarea
          id="problem_statement"
          value={form.problem_statement}
          onChange={(e) => updateField("problem_statement", e.target.value)}
          placeholder="What problem are you solving?"
          rows={3}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="context_notes">Context Notes</Label>
        <Textarea
          id="context_notes"
          value={form.context_notes}
          onChange={(e) => updateField("context_notes", e.target.value)}
          placeholder="Additional context for AI assistance..."
          rows={3}
        />
      </div>
      <Button onClick={handleSave} disabled={saving}>
        {saving ? (
          <Loader2 className="h-4 w-4 animate-spin mr-2" />
        ) : startup ? (
          <Save className="h-4 w-4 mr-2" />
        ) : (
          <Plus className="h-4 w-4 mr-2" />
        )}
        {startup ? "Save Startup" : "Create Startup"}
      </Button>
    </div>
  );
}
