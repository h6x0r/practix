# KODLA: Issues Analysis & Solutions

**Date:** December 2024
**Status:** Analysis Complete

---

## Summary

| # | Issue | Severity | Effort | Files Affected |
|---|-------|----------|--------|----------------|
| 1 | Dashboard Recent Activity infinite list | High | Medium | DashboardPage.tsx, dashboardService.ts |
| 2 | My Tasks page stale data | High | Medium | MyTasksPage.tsx, storage.ts |
| 3 | Courses Resume button stale | High | Medium | CoursesPage.tsx, storage.ts |
| 4 | Roadmap cards outdated design | Medium | High | RoadmapPage.tsx, roadmapService.ts |
| 5 | Analytics yearly grid no month labels | Low | Low | AnalyticsPage.tsx |
| 6 | Settings redundant Editor tab | Low | Low | SettingsPage.tsx |
| 7 | Settings avatar upload broken | Medium | Medium | SettingsPage.tsx, authService.ts |
| 8 | Auth single-device enforcement | High | High | auth.service.ts, auth.guard.ts |

---

## Issue 1: Dashboard Recent Activity Infinite List

### Problem
Recent activity section shows old tasks that persist even after container restarts. Tasks accumulate without limit.

### Root Cause
```typescript
// DashboardPage.tsx:31
taskService.getRecentSubmissions(15) // Fetches last 15 submissions from API
```

The API correctly returns recent submissions, but the issue is:
1. **localStorage persistence** - Started courses/completed tasks in localStorage don't sync with DB
2. **No pagination/limit enforcement** on the UI side
3. **Fixed height container** (380px) but content can overflow

### Current Code
```typescript
// DashboardPage.tsx:184
<div className="bg-white ... h-[380px]">
  <div className="overflow-y-auto p-4 space-y-4">
    {pendingSubmissions.slice(0, 5).map(...)} // Limited to 5
    {passedSubmissions.slice(0, 5).map(...)}  // Limited to 5
  </div>
</div>
```

### Solution
1. **Backend**: Ensure `getRecentSubmissions` filters by `userId` and limits properly
2. **Frontend**: Already slices to 5 items per section - verify API returns fresh data
3. **Clear stale localStorage** when user logs in/out or when API data doesn't match

### Implementation Steps
1. Add `clearStaleData()` method to storage.ts
2. Call on login/logout to reset localStorage submissions cache
3. Add "Clear All" button in Recent Activity header
4. Consider moving `startedCourses` and `completedTasks` to backend

---

## Issue 2: My Tasks Page Stale Data

### Problem
Courses don't reset and always show "Continue Learning" even after container restart. Old tasks persist.

### Root Cause
```typescript
// MyTasksPage.tsx:20-21
const allCourses = await courseService.getAllCourses();
const savedIds = storage.getStartedCourses(); // <-- localStorage!
const filtered = allCourses.filter(c => savedIds.includes(c.id));
```

The page uses `localStorage` for started courses, which:
1. Persists across container restarts
2. Never syncs with backend user progress
3. Can contain stale course IDs

### Solution
Move started courses tracking to backend:

```typescript
// Backend: GET /users/me/started-courses
interface StartedCourse {
  courseId: string;
  startedAt: Date;
  lastAccessedAt: Date;
  progress: number;
}

// Frontend: Use API instead of localStorage
const fetchMyCourses = async () => {
  const startedCourses = await userService.getStartedCourses(); // API call
  setMyCourses(startedCourses);
};
```

### Implementation Steps
1. Add `UserCourse` model to Prisma schema
2. Create `POST /users/me/courses/:courseId/start` endpoint
3. Create `GET /users/me/courses` endpoint
4. Update `MyTasksPage.tsx` to use API
5. Update `CoursesPage.tsx` to use API
6. Keep localStorage as offline fallback only

---

## Issue 3: Courses Resume Button Stale

### Problem
Same as Issue 2 - Resume button shows instead of "Start Learning" because localStorage persists.

### Root Cause
```typescript
// CoursesPage.tsx:38-39
useEffect(() => {
  const saved = storage.getStartedCourses();
  setStartedCourses(saved);
}, []);
```

### Solution
Same as Issue 2 - migrate to backend tracking.

### Quick Fix (Temporary)
Add a "Reset Progress" option or sync localStorage with backend on mount:

```typescript
useEffect(() => {
  // Sync localStorage with backend
  userService.getStartedCourses().then(backendCourses => {
    const backendIds = backendCourses.map(c => c.courseId);
    storage.setStartedCourses(backendIds);
    setStartedCourses(backendIds);
  });
}, []);
```

---

## Issue 4: Roadmap Cards Outdated Design

### Problem
Roadmap cards have old design, not flexible for new course categories:
- Go Backend
- Java Backend
- Algorithms & DS
- Design Patterns
- AI/ML (future)
- Data/Prompt Engineering (future)

### Current Categories
```typescript
// RoadmapPage.tsx:241-248
options: [
  { id: 'backend-go', label: 'Go Backend' },
  { id: 'backend-java', label: 'Java Backend' },
  { id: 'python-data', label: 'Python & Data' },
  { id: 'ai-ml', label: 'AI & Machine Learning' },
  { id: 'software-design', label: 'Software Design' },
  { id: 'algorithms', label: 'Algorithms & DS' },
  { id: 'fullstack', label: 'Fullstack' },
]
```

### Issues
1. Categories don't map 1:1 to actual courses
2. Phase cards lack course-specific styling
3. No dynamic loading from course catalog
4. Limited step types (`task`, `module`, `course`, `external`)

### Solution
1. **Dynamic Category Loading**: Fetch categories from course catalog
2. **Course-Aware Roadmap**: Link roadmap steps to actual course/module/task slugs
3. **Flexible Card Design**: Support different step types with icons:
   - `task` - Code icon
   - `module` - Folder icon
   - `course` - Book icon
   - `reading` - Document icon
   - `project` - Rocket icon

### Implementation Steps
1. Create `RoadmapCategory` model linked to courses
2. Update wizard to load categories dynamically
3. Redesign step cards with better visual hierarchy
4. Add "View Course" button for course-type steps

---

## Issue 5: Analytics Yearly Grid No Month Labels

### Problem
Yearly contributions grid is a flat 364-cell grid without month labels.

### Current Code
```typescript
// AnalyticsPage.tsx:175-189
<div className="grid grid-cols-[repeat(53,1fr)] gap-1">
  {yearlyData.slice(0, 364).map((day, i) => (
    <div className={`w-full aspect-square rounded-[2px] ...`}></div>
  ))}
</div>
```

### Solution
Group by months with labels above each column group:

```typescript
// Proposed structure
<div className="space-y-1">
  {/* Month Labels */}
  <div className="flex gap-1 text-xs text-gray-400 mb-2">
    {months.map(month => (
      <div key={month.name} style={{ width: month.weeks * cellWidth }}>
        {month.name}
      </div>
    ))}
  </div>

  {/* Grid with week columns */}
  <div className="flex gap-1">
    {weeks.map((week, weekIdx) => (
      <div key={weekIdx} className="flex flex-col gap-1">
        {week.days.map(day => (
          <Cell key={day.date} day={day} />
        ))}
      </div>
    ))}
  </div>
</div>
```

### Implementation Steps
1. Restructure `yearlyData` into weeks (7 days per column)
2. Add month labels row
3. Align grid to start on Sunday/Monday
4. Consider GitHub-style layout (7 rows x 53 columns)

---

## Issue 6: Settings Redundant Editor Tab

### Problem
"Editor & Appearance" tab exists in Settings, but same settings already exist in CodeEditorPanel dropdown.

### Current Locations
1. **Settings Page** (`SettingsPage.tsx:177-223`)
   - Font size slider
   - Minimap toggle
   - Vim mode toggle
   - Line numbers toggle

2. **Code Editor** (`EditorSettingsDropdown.tsx`)
   - Same settings available inline

### Solution
**Option A**: Remove from Settings page entirely
- Pros: No duplication, simpler settings
- Cons: Users expect settings in Settings page

**Option B**: Keep in Settings but sync with editor
- Pros: Central location for preferences
- Cons: Two places to manage

### Recommendation
Remove "Editor & Appearance" tab from Settings page. Users can configure editor directly in the workspace.

### Implementation Steps
1. Remove `preferences` tab from Settings sidebar
2. Remove corresponding content section
3. Update sidebar navigation (3 tabs instead of 4)
4. Keep notification and security settings

---

## Issue 7: Settings Avatar Upload Broken

### Problem
1. Max size says "800K" - too large, should be 100-200KB
2. Upload functionality doesn't work
3. No preset avatars option

### Current Code
```typescript
// SettingsPage.tsx:148-155
<div className="flex items-center gap-6">
  <img src={user.avatarUrl} className="w-24 h-24 rounded-full ..." />
  <div>
    <button className="...">{tUI('settings.uploadNew')}</button>
    <div className="text-xs text-gray-500 mt-2">{tUI('settings.avatarHelper')}</div>
  </div>
</div>
```

Button doesn't have `onClick` handler!

### Solution
1. **Implement file upload**:
   ```typescript
   const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
     const file = e.target.files?.[0];
     if (!file) return;

     // Validate size (max 200KB)
     if (file.size > 200 * 1024) {
       showToast('Image too large. Max 200KB.', 'error');
       return;
     }

     // Upload to backend
     const formData = new FormData();
     formData.append('avatar', file);
     const result = await authService.uploadAvatar(formData);
     updateUser({ ...user, avatarUrl: result.url });
   };
   ```

2. **Add preset avatars** (alternative):
   ```typescript
   const PRESET_AVATARS = [
     '/avatars/default-1.png',
     '/avatars/default-2.png',
     // ... more options
   ];
   ```

### Implementation Steps
1. Create `POST /users/me/avatar` endpoint
2. Add file input with size validation (200KB max)
3. Add image compression/resize on upload
4. Add preset avatar selection modal
5. Update helper text to "Max size 200KB"

---

## Issue 8: Authentication Single-Device Enforcement

### Problem
One account can be logged in on multiple devices, allowing subscription sharing.

### Current State
```typescript
// auth.service.ts:68-71
private generateToken(user: User) {
  const payload = { email: user.email, sub: user.id };
  return this.jwtService.sign(payload);
}
```
No session tracking - multiple tokens can be active simultaneously.

### Solution
Implement session-based authentication with device tracking:

```prisma
// Prisma schema addition
model Session {
  id          String   @id @default(uuid())
  userId      String
  user        User     @relation(fields: [userId], references: [id])
  token       String   @unique
  deviceInfo  String?  // Browser, OS, etc.
  ipAddress   String?
  createdAt   DateTime @default(now())
  expiresAt   DateTime
  lastActiveAt DateTime @default(now())

  @@index([userId])
}
```

### Implementation Flow
1. **On Login**:
   - Invalidate all existing sessions (or just oldest)
   - Create new session record
   - Return session token

2. **On API Request**:
   - Validate token exists in sessions table
   - Update `lastActiveAt`
   - Reject if session expired or invalidated

3. **On Logout**:
   - Delete session record

### Implementation Steps
1. Add Session model to Prisma schema
2. Create SessionService with CRUD operations
3. Update AuthService to create sessions on login
4. Create auth guard that validates session
5. Add "Active Sessions" view in Settings
6. Optional: Allow multiple devices for Premium users

---

## Priority Order

### Phase 1: Critical Fixes (Week 1)
1. **Issue 2 & 3**: Migrate startedCourses to backend
   - Fixes stale Resume buttons
   - Fixes My Tasks page
   - Single source of truth

2. **Issue 1**: Dashboard Recent Activity
   - Depends on Phase 1 backend changes
   - Clear localStorage on login

### Phase 2: Security (Week 2)
3. **Issue 8**: Single-device auth
   - Prevents subscription sharing
   - Adds session management

### Phase 3: UX Improvements (Week 3)
4. **Issue 5**: Analytics month labels
   - Visual improvement
   - Low effort

5. **Issue 6**: Remove Settings Editor tab
   - Reduces confusion
   - Low effort

6. **Issue 7**: Fix avatar upload
   - Better UX
   - Medium effort

### Phase 4: Roadmap Redesign (Week 4+)
7. **Issue 4**: Roadmap cards
   - Larger redesign effort
   - Can be done incrementally

---

## Technical Notes

### localStorage Keys to Migrate
```typescript
// Currently in localStorage (should move to backend)
STORAGE_KEYS.STARTED_COURSES = 'kodla_started_courses'
STORAGE_KEYS.COMPLETED_TASKS = 'kodla_completed_tasks'
STORAGE_KEYS.ROADMAP_PREFS = 'kodla_roadmap_prefs'
```

### Keep in localStorage (UI state only)
```typescript
STORAGE_KEYS.THEME = 'kodla_theme'
STORAGE_KEYS.SIDEBAR_COLLAPSED = 'kodla_sidebar_collapsed'
STORAGE_KEYS.TASK_CODE_PREFIX = 'kodla_task_code_' // Draft code
```

### Backend Endpoints Needed
```
POST /users/me/courses/:courseId/start
GET  /users/me/courses
POST /users/me/avatar
GET  /users/me/sessions
DELETE /users/me/sessions/:id
POST /auth/logout (invalidate session)
```
