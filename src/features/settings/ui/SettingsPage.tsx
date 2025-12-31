
import React, { useState, useContext, useEffect, useRef } from 'react';
import { AuthContext } from '@/components/Layout';
import { useToast } from '@/components/Toast';
import { IconCheck } from '@/components/Icons';
import { authService } from '@/features/auth/api/authService';
import { useUITranslation } from '@/contexts/LanguageContext';
import { AuthRequiredOverlay } from '@/components/AuthRequiredOverlay';

// Preset avatars for users who don't want to upload
const PRESET_AVATARS = [
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Felix',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Luna',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Max',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Mia',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Oscar',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Bella',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Charlie',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Ruby',
];

const SettingsPage = () => {
  const { user, updateUser } = useContext(AuthContext);
  const { showToast } = useToast();
  const { tUI } = useUITranslation();
  const [activeTab, setActiveTab] = useState('profile');
  const [isLoading, setIsLoading] = useState(false);
  const [showAvatarPicker, setShowAvatarPicker] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [notifications, setNotifications] = useState({
    emailDigest: true,
    newCourses: true,
    marketing: false,
    securityAlerts: true
  });

  useEffect(() => {
    if (user?.preferences) {
        const p = user.preferences;
        setNotifications(p.notifications);
    }
  }, [user]);

  const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !user) return;

    // Validate size (max 200KB)
    if (file.size > 200 * 1024) {
      showToast(tUI('settings.avatarTooLarge') || 'Image too large. Max 200KB.', 'error');
      return;
    }

    // Validate type
    if (!file.type.startsWith('image/')) {
      showToast(tUI('settings.avatarInvalidType') || 'Please select an image file.', 'error');
      return;
    }

    setIsLoading(true);
    try {
      // Convert to base64 for now (can be changed to FormData upload later)
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = reader.result as string;
        const updatedUser = await authService.updateAvatar(base64);
        updateUser(updatedUser);
        showToast(tUI('settings.avatarUpdated') || 'Avatar updated!', 'success');
        setIsLoading(false);
      };
      reader.readAsDataURL(file);
    } catch (e) {
      showToast(tUI('settings.avatarUpdateFailed') || 'Failed to update avatar.', 'error');
      setIsLoading(false);
    }
  };

  const handlePresetAvatar = async (avatarUrl: string) => {
    if (!user) return;
    setIsLoading(true);
    try {
      const updatedUser = await authService.updateAvatar(avatarUrl);
      updateUser(updatedUser);
      showToast(tUI('settings.avatarUpdated') || 'Avatar updated!', 'success');
      setShowAvatarPicker(false);
    } catch (e) {
      showToast(tUI('settings.avatarUpdateFailed') || 'Failed to update avatar.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    if (!user) return;
    setIsLoading(true);

    try {
        const updatedPrefs = {
            notifications: notifications
        };

        const updatedUser = await authService.updatePreferences(updatedPrefs);
        updateUser(updatedUser);

        showToast(tUI('settings.savedSuccess'), 'success');
    } catch (e) {
        showToast(tUI('settings.savedFailed'), 'error');
    } finally {
        setIsLoading(false);
    }
  };

  interface ToggleProps {
    label: string;
    checked: boolean;
    onChange: (value: boolean) => void;
    description?: string;
  }

  const Toggle = ({ label, checked, onChange, description }: ToggleProps) => (
    <div className="flex items-center justify-between py-4 border-b border-gray-100 dark:border-dark-border last:border-0">
        <div>
            <div className="text-sm font-bold text-gray-900 dark:text-white">{label}</div>
            {description && <div className="text-xs text-gray-500 mt-0.5">{description}</div>}
        </div>
        <button 
            onClick={() => onChange(!checked)}
            className={`w-11 h-6 flex items-center rounded-full transition-colors duration-300 ${checked ? 'bg-brand-600' : 'bg-gray-200 dark:bg-gray-700'}`}
        >
            <div className={`w-4 h-4 bg-white rounded-full shadow-md transform transition-transform duration-300 ${checked ? 'translate-x-6' : 'translate-x-1'}`}></div>
        </button>
    </div>
  );

  if (!user) {
    return (
      <AuthRequiredOverlay
        title={tUI('settings.loginRequired')}
        description={tUI('settings.loginRequiredDesc')}
      >
        {/* Mock Settings Preview */}
        <div className="max-w-4xl mx-auto space-y-8 pb-12">
          <div>
            <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('settings.title')}</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-2">{tUI('settings.description')}</p>
          </div>
          <div className="flex flex-col md:flex-row gap-8 items-start">
            <div className="w-full md:w-64 flex-shrink-0 space-y-1">
              {[
                { id: 'profile', label: tUI('settings.publicProfile'), icon: '👤' },
                { id: 'notifications', label: tUI('settings.notifications'), icon: '🔔' },
                { id: 'security', label: tUI('settings.passwordSecurity'), icon: '🔒' }
              ].map(tab => (
                <div
                  key={tab.id}
                  className={`w-full text-left px-4 py-3 rounded-xl text-sm font-bold flex items-center gap-3 ${
                    tab.id === 'profile'
                      ? 'bg-white dark:bg-dark-surface text-brand-600 shadow-sm border border-gray-100 dark:border-dark-border ring-1 ring-brand-500/10'
                      : 'text-gray-500'
                  }`}
                >
                  <span>{tab.icon}</span>
                  {tab.label}
                </div>
              ))}
            </div>
            <div className="flex-1 w-full bg-white dark:bg-dark-surface p-6 md:p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white border-b border-gray-100 dark:border-dark-border pb-4">{tUI('settings.publicProfile')}</h2>
              <div className="space-y-4 mt-6">
                <div className="flex items-center gap-6">
                  <div className="w-24 h-24 rounded-full bg-gray-200 dark:bg-dark-bg animate-pulse" />
                  <div className="space-y-2">
                    <div className="h-8 w-24 bg-gray-100 dark:bg-dark-bg rounded-lg" />
                    <div className="h-4 w-32 bg-gray-100 dark:bg-dark-bg rounded" />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                  <div className="h-12 bg-gray-50 dark:bg-dark-bg rounded-xl" />
                  <div className="h-12 bg-gray-50 dark:bg-dark-bg rounded-xl" />
                  <div className="h-24 bg-gray-50 dark:bg-dark-bg rounded-xl col-span-2" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </AuthRequiredOverlay>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
       <div>
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('settings.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">{tUI('settings.description')}</p>
      </div>

      <div className="flex flex-col md:flex-row gap-8 items-start">
        {/* Sidebar Nav */}
        <div className="w-full md:w-64 flex-shrink-0 space-y-1">
          {[
            { id: 'profile', label: tUI('settings.publicProfile'), icon: '👤' },
            { id: 'notifications', label: tUI('settings.notifications'), icon: '🔔' },
            { id: 'security', label: tUI('settings.passwordSecurity'), icon: '🔒' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full text-left px-4 py-3 rounded-xl text-sm font-bold flex items-center gap-3 transition-all ${
                activeTab === tab.id 
                  ? 'bg-white dark:bg-dark-surface text-brand-600 shadow-sm border border-gray-100 dark:border-dark-border ring-1 ring-brand-500/10' 
                  : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-dark-border/50 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content Area */}
        <div className="flex-1 w-full bg-white dark:bg-dark-surface p-6 md:p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
           
           {/* --- PROFILE TAB --- */}
           {activeTab === 'profile' && (
             <div className="space-y-8 animate-fade-in">
               <h2 className="text-xl font-bold text-gray-900 dark:text-white border-b border-gray-100 dark:border-dark-border pb-4">{tUI('settings.publicProfile')}</h2>

               <div className="space-y-4">
                 <div className="flex items-center gap-6">
                   <img src={user.avatarUrl || PRESET_AVATARS[0]} className="w-24 h-24 rounded-full ring-4 ring-gray-50 dark:ring-dark-bg object-cover" alt="avatar" />
                   <div className="space-y-2">
                     <div className="flex gap-2">
                       <input
                         ref={fileInputRef}
                         type="file"
                         accept="image/*"
                         onChange={handleAvatarUpload}
                         className="hidden"
                       />
                       <button
                         onClick={() => fileInputRef.current?.click()}
                         disabled={isLoading}
                         className="px-4 py-2 bg-white dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-lg text-sm font-bold text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-dark-border transition-colors shadow-sm disabled:opacity-50"
                       >
                         {tUI('settings.uploadNew')}
                       </button>
                       <button
                         onClick={() => setShowAvatarPicker(!showAvatarPicker)}
                         className="px-4 py-2 bg-brand-50 dark:bg-brand-900/20 border border-brand-200 dark:border-brand-800 rounded-lg text-sm font-bold text-brand-700 dark:text-brand-300 hover:bg-brand-100 dark:hover:bg-brand-900/30 transition-colors"
                       >
                         {tUI('settings.chooseAvatar') || 'Choose Avatar'}
                       </button>
                     </div>
                     <div className="text-xs text-gray-500">{tUI('settings.avatarHelperNew') || 'Max 200KB. PNG, JPG, GIF'}</div>
                   </div>
                 </div>

                 {/* Preset Avatar Picker */}
                 {showAvatarPicker && (
                   <div className="p-4 bg-gray-50 dark:bg-dark-bg rounded-xl border border-gray-200 dark:border-dark-border">
                     <div className="text-xs font-bold text-gray-500 uppercase mb-3">{tUI('settings.presetAvatars') || 'Choose a preset avatar'}</div>
                     <div className="grid grid-cols-4 gap-3">
                       {PRESET_AVATARS.map((avatar, i) => (
                         <button
                           key={i}
                           onClick={() => handlePresetAvatar(avatar)}
                           disabled={isLoading}
                           className={`w-16 h-16 rounded-full overflow-hidden ring-2 ring-transparent hover:ring-brand-500 transition-all ${user.avatarUrl === avatar ? 'ring-brand-500' : ''}`}
                         >
                           <img src={avatar} alt={`Avatar ${i + 1}`} className="w-full h-full object-cover" />
                         </button>
                       ))}
                     </div>
                   </div>
                 )}
               </div>

               <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 <div>
                   <label className="block text-xs font-bold text-gray-500 uppercase mb-2">{tUI('settings.fullName')}</label>
                   <input type="text" defaultValue={user.name} className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-brand-500 outline-none dark:text-white" />
                 </div>
                 <div>
                   <label className="block text-xs font-bold text-gray-500 uppercase mb-2">{tUI('settings.username')}</label>
                   <input type="text" defaultValue="alex_dev" className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-brand-500 outline-none dark:text-white" />
                 </div>
                 <div className="col-span-1 md:col-span-2">
                   <label className="block text-xs font-bold text-gray-500 uppercase mb-2">{tUI('settings.bio')}</label>
                   <textarea rows={4} className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-brand-500 outline-none dark:text-white" placeholder={tUI('settings.bioPlaceholder')}></textarea>
                   <div className="text-right text-xs text-gray-400 mt-1">0/200</div>
                 </div>
               </div>
             </div>
           )}

           {/* --- NOTIFICATIONS TAB --- */}
           {activeTab === 'notifications' && (
              <div className="space-y-8 animate-fade-in">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white border-b border-gray-100 dark:border-dark-border pb-4">{tUI('settings.emailPrefs')}</h2>
                <div className="space-y-1">
                    <Toggle
                        label={tUI('settings.weeklyDigest')}
                        description={tUI('settings.weeklyDigestDesc')}
                        checked={notifications.emailDigest}
                        onChange={(v: boolean) => setNotifications({...notifications, emailDigest: v})}
                    />
                    <Toggle
                        label={tUI('settings.newCourses')}
                        description={tUI('settings.newCoursesDesc')}
                        checked={notifications.newCourses}
                        onChange={(v: boolean) => setNotifications({...notifications, newCourses: v})}
                    />
                    <Toggle
                        label={tUI('settings.marketing')}
                        description={tUI('settings.marketingDesc')}
                        checked={notifications.marketing}
                        onChange={(v: boolean) => setNotifications({...notifications, marketing: v})}
                    />
                     <Toggle
                        label={tUI('settings.securityAlerts')}
                        description={tUI('settings.securityAlertsDesc')}
                        checked={notifications.securityAlerts}
                        onChange={(v: boolean) => setNotifications({...notifications, securityAlerts: v})}
                    />
                </div>
              </div>
           )}

           {/* --- SECURITY TAB --- */}
           {activeTab === 'security' && (
             <div className="space-y-8 animate-fade-in">
               <h2 className="text-xl font-bold text-gray-900 dark:text-white border-b border-gray-100 dark:border-dark-border pb-4">{tUI('settings.security')}</h2>

               <div className="space-y-4">
                 <div>
                   <label className="block text-xs font-bold text-gray-500 uppercase mb-2">{tUI('settings.currentPassword')}</label>
                   <input type="password" className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm outline-none dark:text-white focus:ring-2 focus:ring-brand-500" placeholder="••••••••" />
                 </div>
                 <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-2">{tUI('settings.newPassword')}</label>
                        <input type="password" className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm outline-none dark:text-white focus:ring-2 focus:ring-brand-500" />
                    </div>
                    <div>
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-2">{tUI('settings.confirmPassword')}</label>
                        <input type="password" className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm outline-none dark:text-white focus:ring-2 focus:ring-brand-500" />
                    </div>
                 </div>
                 <button className="px-4 py-2 bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-300 text-xs font-bold rounded-lg border border-gray-200 dark:border-dark-border hover:bg-gray-200 dark:hover:bg-dark-border transition-colors">
                     {tUI('settings.updatePassword')}
                 </button>
               </div>

               <div className="pt-6 border-t border-gray-100 dark:border-dark-border">
                   <div className="flex items-center justify-between">
                       <div>
                           <div className="text-sm font-bold text-gray-900 dark:text-white">{tUI('settings.twoFactor')}</div>
                           <div className="text-xs text-gray-500 mt-1">{tUI('settings.twoFactorDesc')}</div>
                       </div>
                       <button
                         className="px-4 py-2 rounded-lg text-xs font-bold transition-colors bg-green-600 text-white hover:bg-green-700"
                       >
                           {tUI('settings.enable2FA')}
                       </button>
                   </div>
               </div>
             </div>
           )}
           
           {/* Footer Actions */}
           <div className="pt-8 mt-8 border-t border-gray-100 dark:border-dark-border flex justify-end gap-3">
             <button
                onClick={() => window.location.reload()}
                className="px-6 py-2.5 bg-white dark:bg-dark-bg border border-gray-200 dark:border-dark-border text-gray-700 dark:text-gray-300 font-bold rounded-xl hover:bg-gray-50 dark:hover:bg-dark-border transition-colors"
             >
                {tUI('common.cancel')}
             </button>
             <button
                onClick={handleSave}
                disabled={isLoading}
                className="px-8 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5 flex items-center gap-2"
             >
                {isLoading ? <span className="animate-spin">⟳</span> : <IconCheck className="w-4 h-4" />}
                {tUI('common.saveChanges')}
             </button>
           </div>

        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
