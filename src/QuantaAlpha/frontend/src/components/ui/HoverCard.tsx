import * as React from "react"

const HoverCard: React.FC<any> = ({ children }) => <>{children}</>
const HoverCardTrigger: React.FC<any> = ({ children }) => <>{children}</>
const HoverCardContent: React.FC<any> = ({ children, className }) => <div className={className}>{children}</div>

export { HoverCard, HoverCardTrigger, HoverCardContent }
