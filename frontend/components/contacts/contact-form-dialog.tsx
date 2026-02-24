"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { Contact } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";

const CONTACT_TYPES = ["investor", "advisor", "customer", "partner", "team_member", "mentor", "service_provider", "other"];
const STRENGTHS = ["cold", "warm", "hot", "close"];

interface ContactFormDialogProps {
  open: boolean;
  onClose: () => void;
  contact?: Contact | null;
  onSaved: () => void;
}

export function ContactFormDialog({ open, onClose, contact, onSaved }: ContactFormDialogProps) {
  const { toast } = useToast();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [contactType, setContactType] = useState("other");
  const [company, setCompany] = useState("");
  const [title, setTitle] = useState("");
  const [linkedinUrl, setLinkedinUrl] = useState("");
  const [twitterUrl, setTwitterUrl] = useState("");
  const [strength, setStrength] = useState("warm");
  const [tags, setTags] = useState("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (contact) {
      setName(contact.name);
      setEmail(contact.email || "");
      setPhone(contact.phone || "");
      setContactType(contact.contact_type);
      setCompany(contact.company || "");
      setTitle(contact.title || "");
      setLinkedinUrl(contact.linkedin_url || "");
      setTwitterUrl(contact.twitter_url || "");
      setStrength(contact.relationship_strength);
      setTags(contact.tags?.join(", ") || "");
      setNotes(contact.notes || "");
    } else {
      setName(""); setEmail(""); setPhone(""); setContactType("other");
      setCompany(""); setTitle(""); setLinkedinUrl(""); setTwitterUrl("");
      setStrength("warm"); setTags(""); setNotes("");
    }
  }, [contact, open]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    try {
      const body = {
        name: name.trim(),
        email: email.trim() || undefined,
        phone: phone.trim() || undefined,
        contact_type: contactType,
        company: company.trim() || undefined,
        title: title.trim() || undefined,
        linkedin_url: linkedinUrl.trim() || undefined,
        twitter_url: twitterUrl.trim() || undefined,
        relationship_strength: strength,
        tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
        notes: notes.trim() || undefined,
      };
      if (contact) {
        await apiFetch(`/contacts/${contact.id}`, { method: "PATCH", body: JSON.stringify(body) });
        toast({ title: "Contact updated", description: "Your changes have been saved." });
      } else {
        await apiFetch("/contacts/", { method: "POST", body: JSON.stringify(body) });
        toast({ title: "Contact created", description: "New contact has been added." });
      }
      onSaved();
      onClose();
    } catch (err) {
      toast({ title: "Error", description: err instanceof Error ? err.message : "Something went wrong", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{contact ? "Edit Contact" : "New Contact"}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Name</Label>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Jane Smith" />
            </div>
            <div className="flex-1">
              <Label>Company</Label>
              <Input value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Acme Inc" />
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Email</Label>
              <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div className="flex-1">
              <Label>Phone</Label>
              <Input value={phone} onChange={(e) => setPhone(e.target.value)} />
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>Type</Label>
              <Select value={contactType} onValueChange={setContactType}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {CONTACT_TYPES.map((t) => (
                    <SelectItem key={t} value={t}>{t.replace(/_/g, " ")}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <Label>Relationship</Label>
              <Select value={strength} onValueChange={setStrength}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {STRENGTHS.map((s) => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div>
            <Label>Title / Role</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Partner, CTO, etc." />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <Label>LinkedIn</Label>
              <Input value={linkedinUrl} onChange={(e) => setLinkedinUrl(e.target.value)} placeholder="https://linkedin.com/in/..." />
            </div>
            <div className="flex-1">
              <Label>Twitter</Label>
              <Input value={twitterUrl} onChange={(e) => setTwitterUrl(e.target.value)} placeholder="https://twitter.com/..." />
            </div>
          </div>
          <div>
            <Label>Tags (comma-separated)</Label>
            <Input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="vc, seed, ai" />
          </div>
          <div>
            <Label>Notes</Label>
            <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} />
          </div>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={submitting || !name.trim()}>
              {submitting ? "Saving..." : contact ? "Update" : "Create"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
