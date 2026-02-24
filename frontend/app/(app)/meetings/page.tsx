"use client";

import { useState } from "react";
import { Meeting, MeetingTemplate } from "@/lib/types";
import { MeetingList } from "@/components/meetings/meeting-list";
import { MeetingDetail } from "@/components/meetings/meeting-detail";
import { MeetingFormDialog } from "@/components/meetings/meeting-form-dialog";
import { MeetingTemplatesPanel } from "@/components/meetings/meeting-templates-panel";

export default function MeetingsPage() {
  const [selectedMeetingId, setSelectedMeetingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editingMeeting, setEditingMeeting] = useState<Meeting | null>(null);
  const [templateToUse, setTemplateToUse] = useState<MeetingTemplate | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  function handleNewMeeting() {
    setEditingMeeting(null);
    setTemplateToUse(null);
    setShowForm(true);
  }

  function handleEdit(meeting: Meeting) {
    setEditingMeeting(meeting);
    setTemplateToUse(null);
    setShowForm(true);
  }

  function handleUseTemplate(template: MeetingTemplate) {
    setEditingMeeting(null);
    setTemplateToUse(template);
    setShowForm(true);
  }

  function handleSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="flex h-full">
      <div className="flex flex-col">
        <MeetingList
          key={refreshKey}
          selectedId={selectedMeetingId}
          onSelect={setSelectedMeetingId}
          onNewMeeting={handleNewMeeting}
        />
        <MeetingTemplatesPanel onUseTemplate={handleUseTemplate} />
      </div>
      <MeetingDetail
        key={selectedMeetingId}
        meetingId={selectedMeetingId}
        onEdit={handleEdit}
        onDeleted={() => { setSelectedMeetingId(null); setRefreshKey((k) => k + 1); }}
      />
      <MeetingFormDialog
        open={showForm}
        onClose={() => setShowForm(false)}
        meeting={editingMeeting}
        template={templateToUse}
        onSaved={handleSaved}
      />
    </div>
  );
}
