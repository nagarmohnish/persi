"use client";

import { ProfileForm } from "@/components/settings/profile-form";
import { StartupForm } from "@/components/settings/startup-form";

export default function SettingsPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto py-8 px-6 space-y-8">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Manage your profile and startup details.
          </p>
        </div>

        <section className="space-y-4">
          <h2 className="text-lg font-semibold border-b border-border pb-2">
            Profile
          </h2>
          <ProfileForm />
        </section>

        <section className="space-y-4">
          <h2 className="text-lg font-semibold border-b border-border pb-2">
            Startup
          </h2>
          <StartupForm />
        </section>
      </div>
    </div>
  );
}
