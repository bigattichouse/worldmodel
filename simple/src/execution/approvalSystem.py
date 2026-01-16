import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from enum import Enum
import logging
import re
from pathlib import Path

from ..utils.config import ApprovalConfig
from ..utils.logging import get_logger

class ActionType(Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write" 
    NETWORK_REQUEST = "network_request"
    SYSTEM_COMMAND = "system_command"
    PACKAGE_INSTALL = "package_install"
    DATABASE_ACCESS = "database_access"
    ENVIRONMENT_MODIFY = "environment_modify"

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ActionRequest:
    action_type: ActionType
    description: str
    details: Dict[str, Any]
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
class ApprovalResult(Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"

@dataclass
class ApprovalDecision:
    result: ApprovalResult
    reason: str
    reviewer: str
    timestamp: float
    expiry: Optional[float] = None
    conditions: Optional[List[str]] = None

class RiskAssessmentEngine:
    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.danger_patterns = [
            r'rm\s+-rf\s+/',
            r'sudo\s+dd\s+if=',
            r'> /dev/sd[a-z]',
            r'curl.*\|.*sh',
            r'wget.*\|.*sh',
            r'eval.*\$\(',
            r'chmod\s+777',
            r'passwd',
            r'useradd',
            r'usermod',
            r'crontab',
            r'systemctl.*disable',
            r'iptables.*DROP',
            r'mkfs\.',
            r'fdisk',
            r'parted',
            r'mysql.*DROP\s+DATABASE'
        ]
        
        self.sensitive_paths = [
            '/etc/passwd',
            '/etc/shadow', 
            '/etc/sudoers',
            '/root',
            '/boot',
            '/dev',
            '/proc',
            '/sys',
            '~/.ssh',
            '~/.aws',
            '~/.gcp'
        ]
        
        self.restricted_domains = [
            'localhost',
            '127.0.0.1',
            '10.',
            '192.168.',
            '172.'
        ]
    
    def assess_risk(self, action: ActionRequest) -> RiskLevel:
        if action.action_type == ActionType.FILE_READ:
            return self._assess_file_read_risk(action)
        elif action.action_type == ActionType.FILE_WRITE:
            return self._assess_file_write_risk(action)
        elif action.action_type == ActionType.SYSTEM_COMMAND:
            return self._assess_command_risk(action)
        elif action.action_type == ActionType.NETWORK_REQUEST:
            return self._assess_network_risk(action)
        elif action.action_type == ActionType.PACKAGE_INSTALL:
            return self._assess_package_risk(action)
        else:
            return RiskLevel.MEDIUM
    
    def _assess_file_read_risk(self, action: ActionRequest) -> RiskLevel:
        path = action.details.get('path', '')
        
        if any(sensitive in path for sensitive in self.sensitive_paths):
            return RiskLevel.CRITICAL
        
        if path.startswith('/etc/') or path.startswith('/root/'):
            return RiskLevel.HIGH
            
        return RiskLevel.LOW
    
    def _assess_file_write_risk(self, action: ActionRequest) -> RiskLevel:
        path = action.details.get('path', '')
        
        if any(sensitive in path for sensitive in self.sensitive_paths):
            return RiskLevel.CRITICAL
            
        if path.startswith(('/etc/', '/usr/', '/opt/', '/var/lib/')):
            return RiskLevel.HIGH
            
        if path.endswith(('.sh', '.py', '.pl', '.rb', '.js')):
            return RiskLevel.MEDIUM
            
        return RiskLevel.LOW
    
    def _assess_command_risk(self, action: ActionRequest) -> RiskLevel:
        command = action.details.get('command', '')
        
        for pattern in self.danger_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return RiskLevel.CRITICAL
        
        high_risk_commands = ['sudo', 'su', 'ssh', 'scp', 'rsync']
        if any(cmd in command.split() for cmd in high_risk_commands):
            return RiskLevel.HIGH
            
        return RiskLevel.MEDIUM
    
    def _assess_network_risk(self, action: ActionRequest) -> RiskLevel:
        url = action.details.get('url', '')
        
        if any(domain in url for domain in self.restricted_domains):
            return RiskLevel.HIGH
            
        return RiskLevel.MEDIUM
    
    def _assess_package_risk(self, action: ActionRequest) -> RiskLevel:
        package = action.details.get('package', '')
        
        suspicious_packages = ['backdoor', 'trojan', 'malware']
        if any(sus in package.lower() for sus in suspicious_packages):
            return RiskLevel.CRITICAL
            
        return RiskLevel.LOW

class UserApprovalInterface:
    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
    async def request_approval(self, action: ActionRequest) -> ApprovalDecision:
        if not self.config.require_user_approval:
            return ApprovalDecision(
                result=ApprovalResult.APPROVED,
                reason="User approval disabled",
                reviewer="system",
                timestamp=asyncio.get_event_loop().time()
            )
        
        print(f"\n🔐 APPROVAL REQUIRED")
        print(f"Action: {action.action_type.value}")
        print(f"Description: {action.description}")
        print(f"Risk Level: {action.risk_level.name}")
        print(f"Details: {action.details}")
        
        while True:
            try:
                response = input("\nApprove this action? (y/n/details): ").strip().lower()
                
                if response in ['y', 'yes']:
                    return ApprovalDecision(
                        result=ApprovalResult.APPROVED,
                        reason="User approved",
                        reviewer="user",
                        timestamp=asyncio.get_event_loop().time()
                    )
                elif response in ['n', 'no']:
                    reason = input("Reason for denial (optional): ").strip()
                    return ApprovalDecision(
                        result=ApprovalResult.DENIED,
                        reason=reason or "User denied",
                        reviewer="user", 
                        timestamp=asyncio.get_event_loop().time()
                    )
                elif response == 'details':
                    print(f"\nDetailed Information:")
                    for key, value in action.details.items():
                        print(f"  {key}: {value}")
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'details' for more information")
                    
            except (EOFError, KeyboardInterrupt):
                return ApprovalDecision(
                    result=ApprovalResult.DENIED,
                    reason="User interrupted approval",
                    reviewer="user",
                    timestamp=asyncio.get_event_loop().time()
                )

class RuleBasedApprover:
    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.auto_approve_patterns = config.auto_approve_patterns or []
        self.auto_deny_patterns = config.auto_deny_patterns or []
        
    def evaluate(self, action: ActionRequest) -> Optional[ApprovalDecision]:
        action_str = f"{action.action_type.value}:{action.description}"
        
        for pattern in self.auto_deny_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                return ApprovalDecision(
                    result=ApprovalResult.DENIED,
                    reason=f"Matches auto-deny pattern: {pattern}",
                    reviewer="rule_engine",
                    timestamp=asyncio.get_event_loop().time()
                )
        
        for pattern in self.auto_approve_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                return ApprovalDecision(
                    result=ApprovalResult.APPROVED,
                    reason=f"Matches auto-approve pattern: {pattern}",
                    reviewer="rule_engine", 
                    timestamp=asyncio.get_event_loop().time()
                )
        
        if action.risk_level == RiskLevel.LOW and self.config.auto_approve_low_risk:
            return ApprovalDecision(
                result=ApprovalResult.APPROVED,
                reason="Low risk auto-approval",
                reviewer="rule_engine",
                timestamp=asyncio.get_event_loop().time()
            )
        
        return None

class ApprovalCache:
    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.cache: Dict[str, ApprovalDecision] = {}
        self.logger = get_logger(__name__)
    
    def _generate_key(self, action: ActionRequest) -> str:
        return f"{action.action_type.value}:{action.description}:{hash(frozenset(action.details.items()))}"
    
    def get_cached_decision(self, action: ActionRequest) -> Optional[ApprovalDecision]:
        if not self.config.cache_approvals:
            return None
            
        key = self._generate_key(action)
        decision = self.cache.get(key)
        
        if decision and decision.expiry:
            current_time = asyncio.get_event_loop().time()
            if current_time > decision.expiry:
                del self.cache[key]
                return None
        
        return decision
    
    def cache_decision(self, action: ActionRequest, decision: ApprovalDecision) -> None:
        if not self.config.cache_approvals:
            return
            
        key = self._generate_key(action)
        
        if self.config.approval_cache_ttl > 0:
            decision.expiry = asyncio.get_event_loop().time() + self.config.approval_cache_ttl
        
        self.cache[key] = decision
        
        if len(self.cache) > 1000:  # Prevent unbounded growth
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]

class ApprovalSystem:
    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.risk_engine = RiskAssessmentEngine(config)
        self.user_interface = UserApprovalInterface(config)
        self.rule_approver = RuleBasedApprover(config) 
        self.cache = ApprovalCache(config)
        
        self.approval_history: List[Dict[str, Any]] = []
        
    async def request_approval(self, action: ActionRequest) -> ApprovalDecision:
        try:
            action.risk_level = self.risk_engine.assess_risk(action)
            
            cached_decision = self.cache.get_cached_decision(action)
            if cached_decision:
                self.logger.info(f"Using cached approval decision: {cached_decision.result.value}")
                self._log_approval(action, cached_decision, from_cache=True)
                return cached_decision
            
            rule_decision = self.rule_approver.evaluate(action)
            if rule_decision:
                self.cache.cache_decision(action, rule_decision)
                self._log_approval(action, rule_decision, from_rules=True)
                return rule_decision
            
            if action.risk_level == RiskLevel.CRITICAL:
                decision = ApprovalDecision(
                    result=ApprovalResult.DENIED,
                    reason="Critical risk level",
                    reviewer="system",
                    timestamp=asyncio.get_event_loop().time()
                )
            else:
                decision = await self.user_interface.request_approval(action)
            
            self.cache.cache_decision(action, decision)
            self._log_approval(action, decision)
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in approval system: {e}")
            return ApprovalDecision(
                result=ApprovalResult.DENIED,
                reason=f"System error: {str(e)}",
                reviewer="system",
                timestamp=asyncio.get_event_loop().time()
            )
    
    def _log_approval(self, action: ActionRequest, decision: ApprovalDecision, 
                     from_cache: bool = False, from_rules: bool = False) -> None:
        entry = {
            'action_type': action.action_type.value,
            'description': action.description,
            'risk_level': action.risk_level.name,
            'decision': decision.result.value,
            'reason': decision.reason,
            'reviewer': decision.reviewer,
            'timestamp': decision.timestamp,
            'from_cache': from_cache,
            'from_rules': from_rules
        }
        
        self.approval_history.append(entry)
        self.logger.info(f"Approval logged: {entry}")
        
        if len(self.approval_history) > 1000:  # Prevent unbounded growth
            self.approval_history = self.approval_history[-500:]
    
    def get_approval_stats(self) -> Dict[str, Any]:
        if not self.approval_history:
            return {}
        
        total_requests = len(self.approval_history)
        approved = sum(1 for entry in self.approval_history if entry['decision'] == 'approved')
        denied = total_requests - approved
        
        from_cache = sum(1 for entry in self.approval_history if entry['from_cache'])
        from_rules = sum(1 for entry in self.approval_history if entry['from_rules'])
        from_user = total_requests - from_cache - from_rules
        
        risk_distribution = {}
        for entry in self.approval_history:
            risk = entry['risk_level']
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        return {
            'total_requests': total_requests,
            'approved': approved,
            'denied': denied,
            'approval_rate': approved / total_requests if total_requests > 0 else 0,
            'from_cache': from_cache,
            'from_rules': from_rules,
            'from_user': from_user,
            'risk_distribution': risk_distribution,
            'recent_decisions': self.approval_history[-10:]
        }

# Convenience functions
async def request_file_read(approver: ApprovalSystem, path: str, description: str = "") -> ApprovalDecision:
    action = ActionRequest(
        action_type=ActionType.FILE_READ,
        description=description or f"Read file: {path}",
        details={'path': path}
    )
    return await approver.request_approval(action)

async def request_file_write(approver: ApprovalSystem, path: str, content: str = "", description: str = "") -> ApprovalDecision:
    action = ActionRequest(
        action_type=ActionType.FILE_WRITE,
        description=description or f"Write file: {path}",
        details={'path': path, 'content_preview': content[:100] + '...' if len(content) > 100 else content}
    )
    return await approver.request_approval(action)

async def request_system_command(approver: ApprovalSystem, command: str, description: str = "") -> ApprovalDecision:
    action = ActionRequest(
        action_type=ActionType.SYSTEM_COMMAND,
        description=description or f"Execute command: {command}",
        details={'command': command}
    )
    return await approver.request_approval(action)

async def request_network_access(approver: ApprovalSystem, url: str, method: str = "GET", description: str = "") -> ApprovalDecision:
    action = ActionRequest(
        action_type=ActionType.NETWORK_REQUEST,
        description=description or f"Network request: {method} {url}",
        details={'url': url, 'method': method}
    )
    return await approver.request_approval(action)